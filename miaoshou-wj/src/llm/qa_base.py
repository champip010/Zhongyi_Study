import asyncio
import json
import re
import uuid
from typing import AsyncIterable, Awaitable
from typing import Optional

from celery import Celery
from langchain_community.chat_models import ChatOpenAI

from langchain_community.embeddings import OpenAIEmbeddings

from llm.ai_llm_utils import get_llm_bean
from llm.extract_handler import llm_extract_meridians, llm_extract_diseases
from llm.semi_chats import (
    get_final_chain, parse_prompt_and_prompt_params, PromptNameConstant,
    identify_language, generate_new_question, llm_estimate_question,
)
from logger import get_logger
from models.chat import (
    update_message_by_id,
    insert_chat_history,
    update_chat_name,
    get_chat_history_by_num,
)
from models.settings import CelerySettings
from models.settings import common_dependencies, common_index, BrainSettings
from utils.redisutils import redis_service
from utils.streaming_aiter import AsyncIteratorCallbackHandler
from vectorstore.paper_embedding import PaperOpensearchVectorStore
from rag.rag_system import MedicalRAG

settings = CelerySettings()
logger = get_logger(__name__)
app_celery = Celery('worker', broker=settings.broker_url)


def _extract_value(json_string: str, key: str):
    pattern = re.compile(rf'"?{key}"?\s*:\s*((\{{.*?\}})|("((?:[^"\\]|\\.)*)"|(\b[^,\s]*\b)))',
                         re.DOTALL)

    match = pattern.search(json_string)
    if match:
        return match.group(1).replace('\\"', '"').replace("\\\\", "\\").strip('"').strip("'")
    else:
        return None


class QABaseBrainPicking:
    text_generation_interval_source = BrainSettings().text_generation_interval_source

    def __init__(
            self,
            chat_id: str,
            user_id: str,
            prompt_group_id: str,
            callback: Optional[AsyncIteratorCallbackHandler],
            llm_stream: Optional[ChatOpenAI],
            llm: Optional[ChatOpenAI]
    ):
        self.rag = MedicalRAG()
        self.user_id = user_id
        self.chat_id = chat_id
        self.prompt_group_id = prompt_group_id
        if callback is None:
            self.callback = AsyncIteratorCallbackHandler()
        if llm_stream is None:
            self.llm_stream = ChatOpenAI(
                model_name=BrainSettings().glm4_model_name,
                temperature=float(parse_prompt_and_prompt_params(
                    prompt_name=PromptNameConstant.FINAL_ANSWER_LLM_TEMPERATURE,
                    prompt_group_id=self.prompt_group_id)[0]),
                openai_api_key=BrainSettings().glm4_openai_api_key,
                openai_api_base=BrainSettings().glm4_openai_api_base,
                streaming=True, callbacks=[self.callback]
            )
        if llm is None:
            self.llm = get_llm_bean(0)

    @property
    def get_embeddings(self) -> OpenAIEmbeddings:
        common = common_dependencies()
        return common['embeddings']

    @property
    def vector_store(self) -> PaperOpensearchVectorStore:
        indexs = common_index()
        return PaperOpensearchVectorStore(
            opensearch_url=BrainSettings().opensearch_url,
            index_name=indexs['vectors'],
            embedding_function=self.get_embeddings,
            is_aoss=False)

    def search_public_docs(self, query, k):
        return self.vector_store.similarity_search(query=query, k=k)

    def search_perform_documents(self, query: str, use_hybrid_retrieval: bool = False):
        # 获取向量库
        search_docs = self.rag.search(query=query, k=BrainSettings().rag_search_k)
        if len(search_docs) == 0:
            logger.info(f"没有找到相关文档")
        return search_docs

    def query_is_estimate_question(self) -> tuple[bool, str]:
        cache_key = f"system:{self.prompt_group_id}:{PromptNameConstant.IS_ESTIMATE_QUESTION}"
        data = redis_service.get_prompt(name=cache_key)
        prompt_content = data['prompt_content']
        prompt_content = json.loads(prompt_content)
        return prompt_content["is_estimate"], prompt_content["estimate_prompt"]

    async def generate_stream(
            self,
            question: str,
            user_id: str,
    ) -> AsyncIterable:
        """
        步骤：
            1. 判断问题的语言
            2. 判断问题是否和健康相关
                - 不相关，返回提示
                - 相关
                    1. 查询向量库
                    2. 回答问题
                        - 有文档，根据有文档的prompt回答问题
                        - 无文档，根据无文档的prompt回答问题
        """
        logger.info(f'生成答案::::')
        history_message_num, history_message_params = parse_prompt_and_prompt_params(
            prompt_name=PromptNameConstant.QUERY_HISTORY_MESSAGE_COUNT, prompt_group_id=self.prompt_group_id)
        history_message = get_chat_history_by_num(self.chat_id, int(history_message_num))
        # 更新历史记录
        streamed_chat_history = insert_chat_history(
            chat_id=self.chat_id,
            user_message=question,
            assistant="",
            only_assistant="",
            user_id=user_id,
        )
        # 判断中英文
        estimate_lan_result = await identify_language(text=question)
        if estimate_lan_result == 0:
            from_str = '简体中文'
        else:
            from_str = '英文'
        logger.info(f'语种判断结果:::{from_str}')
        # 改写问题
        new_question = generate_new_question(question, history_message, self.prompt_group_id)
        logger.info(f'问题改写为:::{new_question}')
        # 是否判断问题
        is_estimate_question, estimate_prompt = self.query_is_estimate_question()
        if is_estimate_question:
            # 判断问题是否和健康相关
            estimate_qa_result = llm_estimate_question(
                question=new_question,
                prompt_name=estimate_prompt,
                prompt_group_id=self.prompt_group_id,
                history_message=history_message
            )
            if estimate_qa_result is False:  # 不相关，则直接返回提示
                if from_str == '简体中文':
                    unrelated_question_answer_prompt, unrelated_question_answer_params = parse_prompt_and_prompt_params(
                        prompt_name=PromptNameConstant.UNRELATED_QUESTION_ANSWER_CN,
                        prompt_group_id=self.prompt_group_id)
                else:
                    unrelated_question_answer_prompt, unrelated_question_answer_params = parse_prompt_and_prompt_params(
                        prompt_name=PromptNameConstant.UNRELATED_QUESTION_ANSWER_EN,
                        prompt_group_id=self.prompt_group_id)
                yield "event: start\ndata: " + "{}" + "\n\n"
                for item_str in unrelated_question_answer_prompt:
                    streamed_chat_history.assistant = item_str
                    id_uuid4 = uuid.uuid4()
                    await asyncio.sleep(self.text_generation_interval_source)
                    yield f"event: message\ndata: {json.dumps(streamed_chat_history.to_dict())}\n\nid:{id_uuid4}\nretry: 5000\n"
                dict_ = {
                    "only_assistant": unrelated_question_answer_prompt
                }
                if len(history_message) == 0:
                    # chat_name = self.first_chat_update_chat_name(question=question,answer=unrelated_question_answer_prompt)
                    update_chat_name(chat_id=self.chat_id, chat_name=question)
                    dict_["chat_name"] = question
                last_iteration_content = f"event: last\ndata: {json.dumps(dict_, ensure_ascii=False)}\n\n"
                yield last_iteration_content
                # 回答+来源
                assistant = unrelated_question_answer_prompt
                # 回答
                only_assistant = unrelated_question_answer_prompt
                # 更新数据库
                update_message_by_id(
                    chat_id=self.chat_id,
                    message_id=streamed_chat_history.message_id,
                    user_message=question,
                    assistant=assistant,
                    only_assistant=only_assistant
                )
                return
        # 不相关或者判断结果是健康问题
        question_in_where_value, doc_chat_with_human_params = parse_prompt_and_prompt_params(
            prompt_name=PromptNameConstant.QUESTION_IN_WHERE, prompt_group_id=self.prompt_group_id)
        # 搜索文档
        search_docs = self.search_perform_documents(query=new_question)
        is_exist_doc = bool(len(search_docs))
        logger.info(f"判断问题是否存在片段: {is_exist_doc}")
        # 其他情况正常回答
        fianl_answer_chain, final_chain_dict = get_final_chain(
            question_in_where=question_in_where_value,
            from_str=from_str,
            prompt_group_id=self.prompt_group_id,
            llm_stream=self.llm_stream,
            llm=self.llm,
            is_exist_doc=is_exist_doc,
            doc_list=search_docs,
            question=new_question
        )
        # 补充历史消息
        final_chain_dict["history_message"] = history_message
        # Initialize a list to hold the tokens
        # 用来存放 回答+来源
        response_tokens = []
        # 用来存放错误消息
        response_error = []

        async def wrap_done(fn: Awaitable, event: asyncio.Event):
            try:
                await fn
            except Exception as e:
                response_error.append(e)
                logger.error(f"Caught exception: {e}")
            finally:
                event.set()

        run = asyncio.create_task(wrap_done(
            # 把问题 聊天历史补全
            fianl_answer_chain.acall(final_chain_dict),
            self.callback.done,
        ))

        first_iteration = True  # 标志用于检查是否是第一次循环

        async for token in self.callback.aiter():
            if first_iteration:
                yield "event: start\ndata: " + "{}" + "\n\n"
                first_iteration = False
                # 请求结束时间
            # 限制不返回code代码块内容
            if token in ['```markdown', 'markdown', '```']:
                logger.info("出现代码块啦: %s", token)
                token = ''
            logger.info("Token: %s", token)
            # Add the token to the response_tokens list
            response_tokens.append(token)
            streamed_chat_history.assistant = token
            id_uuid4 = uuid.uuid4()
            yield f"event: message\ndata: {json.dumps(streamed_chat_history.to_dict())}\n\nid:{id_uuid4}\nretry: 5000\n"

        # 暂存一份回答
        response_assistant = response_tokens.copy()
        is_show_source = BrainSettings().is_show_source
        is_show_img = BrainSettings().is_show_img

        # 判断病症
        content = "".join(response_tokens)
        imgs = []
        if is_show_img:
            diseases_imgs = llm_extract_diseases(question, content, 0)

            # 如果病症空
            meridians_imgs = llm_extract_meridians(question, content, 0.1)

            imgs += diseases_imgs
            imgs += meridians_imgs

            #  判断是否有筋络图片
            if imgs:
                meridian_content = "<div class='meridian'>"
                for img in imgs:
                    # str1 += f'![示例图片]({img} "{"示例图片标题"}")'
                    if 'url' in img:
                        if isinstance(img.get("url", []), list):
                            for img_item in img.get("url", []):
                                meridian_content += f'<img class="meridian_img" src="{img_item}"  alt="{img_item}">'
                        else:
                            meridian_content += f'<img class="meridian_img" src="{img["url"]}"  alt="{img["name"]}">'
                meridian_content += '</div>'

                response_tokens.append(meridian_content)
                meridian_content_json = {"assistant": meridian_content, "message_id": streamed_chat_history.message_id}

                yield "event: message\ndata: " + json.dumps(meridian_content_json) + "\n\nid:" + str(
                    uuid.uuid4()) + "\nretry: 5000\n"

        if is_show_source is True and len(search_docs) != 0:
            response_tokens.append("\n\n")
            data_json_1 = {"assistant": "\n\n",
                           "message_id": streamed_chat_history.message_id
                           }
            yield "event: message\ndata: " + json.dumps(data_json_1) + "\n\nid:" + str(
                uuid.uuid4()) + "\nretry: 5000\n"
            response_tokens.append("出处:")
            data_json_2 = {"assistant": "出处:",
                           "message_id": streamed_chat_history.message_id}
            yield "event: message\ndata: " + json.dumps(data_json_2) + "\n\nid:" + str(
                uuid.uuid4()) + "\nretry: 5000\n"
            # 添加出处
            for index, item in enumerate(search_docs):
                id_uuid4 = uuid.uuid4()
                if "paper_title" in item.metadata:
                    data = "<details><summary>" + str(index + 1) + ". " + str(
                        item.metadata['paper_title']) + "</summary>" + str(item.page_content) + "</details>"
                else:
                    data = "<details><summary>" + str(index + 1) + ". " + str(
                        item.metadata['file_name']) + "</summary>" + str(item.page_content) + "</details>"
                data_json = {"assistant": data,
                             "message_id": streamed_chat_history.message_id
                             }
                response_tokens.append(data)
                await asyncio.sleep(self.text_generation_interval_source)
                yield "event: message\ndata: " + json.dumps(data_json) + "\n\nid:" + str(
                    id_uuid4) + "\nretry: 5000\n"

        # 若是出现错误，把错误返还给前端
        if len(response_error) != 0:
            id_uuid4 = uuid.uuid4()

            assistant = str(response_error[0])
            # token过长问题改写，若是别的错误直接返回原始信息
            if "Please reduce the length of the messages" in assistant:
                assistant = "此主题已达到最大聊天长度限制，请切换主题重新问答。"
            data_json = {"assistant": assistant,
                         "message_id": streamed_chat_history.message_id
                         }
            response_tokens.append(assistant)
            yield f"event: message\ndata: {json.dumps(data_json)}\n\nid:{id_uuid4}\nretry: 5000\n"
        # 回答+来源
        assistant = "".join(response_tokens)
        # 回答
        only_assistant = "".join(response_assistant)
        dict_ = {
            "only_assistant": only_assistant
        }
        # 第一次会话修改chat名字
        if len(history_message) == 0:
            update_chat_name(chat_id=self.chat_id, chat_name=question)
            dict_["chat_name"] = question
            # 循环结束后，添加最后一次循环的内容
        last_iteration_content = f"event: last\ndata: {json.dumps(dict_, ensure_ascii=False)}\n\n"
        yield last_iteration_content
        await run
        # 更新数据库
        update_message_by_id(
            chat_id=self.chat_id,
            message_id=streamed_chat_history.message_id,
            user_message=question,
            assistant=assistant,
            only_assistant=only_assistant
        )
