import re
from enum import Enum

from baby_agi.all_purpose_chain import AllPurposeChain
from llm.ai_llm_utils import get_llm_bean
from llm.semi_chats import estimate_semiconductor_question_result, generate_new_question
from logger import get_logger
from models.chat import update_message_by_id, insert_chat_history, get_chat_history_by_num
from models.prompt_group import PromptGroup
from models.settings import common_dependencies, common_index
from utils.dateutils import get_shanghai_timestamp_ms, timestamp_to_datetime
from utils.redisutils import RedisService, redis_client

logger = get_logger(__name__)
common = common_dependencies()
indexs = common_index()


def _extract_translate_value(json_string: str, key: str = ''):
    pattern = re.compile(r'result: (.+)')
    match = pattern.search(json_string)
    if match:
        return match.group(1).replace('\\"', '"').replace("\\\\", "\\").strip('"').strip("'")
    else:
        print(f"Could not find {key} in {json_string}")
        return json_string


class RedisKey(Enum):
    CHECK_LANGUAGE_PROMPT = "system:{}:language_estimate"  # 检查语言prompt
    IS_CHECK_SEMI_QUESTION = "system:{}:is_estimate_semiconductor_question"  # 判断是否进行半导体检查参数 1检查 0不检查
    IS_SEMI_QUESTION_PROMPT = "system:{}:estimate_semiconductor_question"  # 检查是否是半导体问题prompt
    NOT_RELATION_CN_ANSWER = "system:{}:unrelated_question_answer_cn"  # 不相关问题中文回答
    NOT_RELATION_EN_ANSWER = "system:{}:unrelated_question_answer_en"  # 不相关问英文题回答
    QUESTION_TRANSLATE_PROMPT = "system:{}:language_translate"  # 翻译
    GENERATE_NEW_QUESTION = "system:{}:generate_new_question"  # 生成新问题
    IS_TRANSLATE_DOC = "system:{}:is_translate_doc"  # 是否翻译文档
    RESULT_DOC_CN_SYSTEM_PROMPT = "system:{}:doc_chat_answer_question_in_system_cn"  # 有文档+中文+system
    RESULT_DOC_EN_SYSTEM_PROMPT = "system:{}:doc_chat_answer_question_in_system_en"  # 有文档+英文+system
    RESULT_DOC_CN_HUMAN_PROMPT = "system:{}:doc_chat_answer_question_in_human_cn"  # 有文档+中文+human
    RESULT_DOC_EN_HUMAN_PROMPT = "system:{}:doc_chat_answer_question_in_human_en"  # 有文档+英文+human
    RESULT_NO_DOC_CN_SYSTEM_PROMPT = "system:{}:no_doc_chat_answer_question_in_system_cn"  # 无文档+中文+system
    RESULT_NO_DOC_EN_SYSTEM_PROMPT = "system:{}:no_doc_chat_answer_question_in_system_en"  # 无文档+英文+system
    RESULT_NO_DOC_CN_HUMAN_PROMPT = "system:{}:no_doc_chat_answer_question_in_human_cn"  # 无文档+中文+human
    RESULT_NO_DOC_EN_HUMAN_PROMPT = "system:{}:no_doc_chat_answer_question_in_human_en"  # 无文档+英文+human
    CHUNK_OPTIONS = "system:{}:chunk_options"  # 分块选项


def package_result(**kwargs):
    return {"status": "True", "message": "Success", "data": kwargs}


def get_config_by_name(prompt_name: str):
    dsl = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "prompt_name": prompt_name
                        }
                    }
                ]
            }
        }
    }
    response = common["opensearch"].search(
        body=dsl,
        index=indexs["prompt"]
    )
    result = []
    for item in response['hits']['hits']:
        item['_source']['id'] = item['_id']
        result.append(item['_source'])
    return result


def update_prompt(prompt_name: str, prompt_content, prompt_params: str):
    dep = common_dependencies()
    opensearch_client = dep['opensearch']
    group_id = PromptGroup.get_system_group()[0]
    dsl = {
        "script": {
            "source": """
                            ctx._source.prompt_content = params.prompt_content;
                            ctx._source.prompt_params = params.prompt_params;
                       """,
            "lang": "painless",
            "params": {
                "prompt_content": prompt_content,
                "prompt_params": prompt_params,

            }
        },
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "group_id": group_id
                        }
                    },
                    {
                        "term": {
                            "prompt_name": prompt_name
                        }
                    }
                ]
            }
        }
    }
    print(f'格式化之后的DSL{dsl}')
    response = opensearch_client.update_by_query(index=common_index()["prompt"], body=dsl, timeout=5000, refresh=True)
    print(response)
    if response.get("updated") == 1:
        # redis_client.hgetall(name="system:" + group_id + ":" + prompt_name)
        redis_client.delete("system:" + group_id + ":" + prompt_name)
        return {"data": "更新成功", "code": "200"}
    else:
        return {"data": "更新失败", "code": "400"}

def update_config(prompt_name: str, prompt_content, prompt_params: str):
    dep = common_dependencies()
    opensearch_client = dep['opensearch']
    dsl = {
        "script": {
            "source": """
                            ctx._source.value = params.value;
                       """,
            "lang": "painless",
            "params": {
                "value": prompt_content,
            }
        },
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "name": prompt_name
                        }
                    }
                ]
            }
        }
    }
    print(f'格式化之后的DSL{dsl}')
    response = opensearch_client.update_by_query(index=common_index()["config"], body=dsl, timeout=5000, refresh=True)
    print(response)
    if response.get("updated") == 1:
        cache_key = f"{'config'}:{prompt_name}"
        redis_client.delete(cache_key)
        return {"data": "更新成功", "code": "200"}
    else:
        return {"data": "更新失败", "code": "400"}


class SemiChain:

    def __init__(self, user_id: str, question='', temperature=0.8, space_type='cosinesimil', search_threshold=1.5,
                 doc_count=4,
                 chat_model='system', is_translate_doc=1, chat_id='', query_history_message_count=5):
        self.language = '中文'
        self.temperature = temperature
        self.group_id = ''
        self.user_id = user_id
        self.question = question
        self.new_question = ''
        self.space_type = space_type
        self.search_threshold = search_threshold
        self.is_translate_doc = is_translate_doc
        self.doc_count = doc_count
        self.chat_model = chat_model
        self.chat_glm_llm = get_llm_bean(temperature)
        # 获取系统group组
        self.get_group_id()
        self.common_glm_llm = get_llm_bean(0.8)
        # 判断语种
        self.identify_language()
        self.total_time = 0,  # 总时间
        self.language_estimate_time = 0  # 判断问题语种的时间
        self.estimate_semiconductor_question_time = 0  # 判断半导体所相关问题的时间
        self.generate_new_question_time = 0  # 生成新问题的时间
        self.language_translate_question_time = 0  # 翻译问题的时间
        self.search_doc_time = 0  # 搜索文档时间
        self.language_translate_doc_time = 0  # 翻译文档时间
        self.final_chain_time = 0  # 最后问答链的时间
        self.chat_id = chat_id  # chat id
        self.query_history_message_count = query_history_message_count  # 查询历史消息的次数

    def get_group_id(self):

        group_id = redis_client.get("system_group_id")
        if not group_id:
            self.group_id = PromptGroup.get_system_group()[0]
            redis_client.setex("system_group_id", 24 * 60 * 60, self.group_id)
        else:
            self.group_id = group_id

    def check_language(self):
        prompt_obj = RedisService.get_prompt(RedisKey.CHECK_LANGUAGE_PROMPT.value.format(self.group_id))
        chain = AllPurposeChain.universal_llm_adapter(llm=self.common_glm_llm, prompt=prompt_obj['prompt_content'],
                                                      parameter=prompt_obj['prompt_params'].split(','))
        check_language_start_time = get_shanghai_timestamp_ms()
        logger.info(f'判断语种的开始时间:::{timestamp_to_datetime(check_language_start_time)}')
        resp = chain.run({'question': self.question})
        self.language = _extract_translate_value(resp, 'language')
        check_language_end_time = get_shanghai_timestamp_ms()
        logger.info(f'判断语种的结束时间:::{timestamp_to_datetime(check_language_end_time)}')

    # 判断text中英文，返回中文，返回英文
    def identify_language(self):

        check_language_start_time = get_shanghai_timestamp_ms()
        logger.info(f'判断语种的开始时间:::{timestamp_to_datetime(check_language_start_time)}')

        ##########################################################
        # 使用正则表达式匹配中文字符，中文编码范围一般为\u4e00到\u9fff
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        # 使用正则表达式匹配英文单词，假设单词由字母数字及下划线组成
        english_pattern = re.compile(r'\b\w+\b')

        # 统计中文字符数量
        chinese_characters = re.findall(chinese_pattern, self.question)
        chinese_count = len(chinese_characters)

        # 统计英文单词数量
        english_words = re.findall(english_pattern, self.question)
        english_count = sum([1 for word in english_words if re.match(r'^[A-Za-z]+$', word)])
        ##########################################################
        # 判断哪种语言的字符数量更多
        chNum = 2 * chinese_count

        check_language_end_time = get_shanghai_timestamp_ms()
        logger.info(f'判断语种的结束时间:::{timestamp_to_datetime(check_language_end_time)}')
        check_language = check_language_end_time - check_language_start_time
        self.language_estimate_time = check_language
        if chNum > english_count:
            self.language = '中文'
        else:
            self.language = '英文'

    def _get_chat_history(self) -> list:
        history_message = get_chat_history_by_num(self.chat_id, self.query_history_message_count)
        return history_message

    def run(self):

        history_message = self._get_chat_history()
        streamed_chat_history = insert_chat_history(
            chat_id=self.chat_id,
            user_message=self.question,
            assistant="",
            only_assistant="",
            user_id=self.user_id,
        )

        generate_new_question_prompt_content = \
            RedisService.get_prompt(RedisKey.GENERATE_NEW_QUESTION.value.format(self.group_id))[
                'prompt_content']

        requests_start_time = get_shanghai_timestamp_ms()
        logger.info(f'请求的开始时间:::{timestamp_to_datetime(requests_start_time)}')

        # 生成新问题
        generate_new_question_start_timestamp = get_shanghai_timestamp_ms()
        logger.info(f'生成新问题的开始时间:::{timestamp_to_datetime(generate_new_question_start_timestamp)}')
        self.new_question = generate_new_question(self.question, history_message, self.group_id)
        generate_new_question_end_timestamp = get_shanghai_timestamp_ms()
        logger.info(f'生成新问题的结束时间:::{timestamp_to_datetime(generate_new_question_end_timestamp)}')
        self.generate_new_question_time = generate_new_question_end_timestamp - generate_new_question_start_timestamp

        # 是否开启半所问题判断
        is_check_relation = RedisService.get_prompt(RedisKey.IS_CHECK_SEMI_QUESTION.value.format(self.group_id))

        if is_check_relation['prompt_content'] == '1':
            # 开启了判断
            estimate_question_start_timestamp = get_shanghai_timestamp_ms()
            logger.info(f'发往Azure的开始时间:::{timestamp_to_datetime(estimate_question_start_timestamp)}')

            estimate_question_result = estimate_semiconductor_question_result(question=self.new_question,
                                                                              prompt_group_id=self.group_id,
                                                                              history_message=history_message)
            estimate_question_end_timestamp = get_shanghai_timestamp_ms()
            logger.info(f'发往Azure的结束时间:::{timestamp_to_datetime(estimate_question_end_timestamp)}')
            self.estimate_semiconductor_question_time = estimate_question_end_timestamp - estimate_question_start_timestamp

            # 按照语种回答
            if estimate_question_result == 0:
                requests_end_time = get_shanghai_timestamp_ms()
                logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                self.total_time = requests_end_time - requests_start_time
                if "中文" in self.language:
                    unrelated_question_answer_cn = \
                        RedisService.get_prompt(RedisKey.NOT_RELATION_CN_ANSWER.value.format(self.group_id))[
                            'prompt_content']
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=unrelated_question_answer_cn,
                        only_assistant=unrelated_question_answer_cn
                    )
                    return {
                        "data": unrelated_question_answer_cn, "doc": [],
                        "generate_new_question_prompt_content": generate_new_question_prompt_content,
                        "generate_new_question": self.new_question,
                        "history_messages": history_message,
                        "chat_id": self.chat_id,
                        'duration':
                            {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,

                            }
                    }
                else:
                    unrelated_question_answer_en = \
                        RedisService.get_prompt(RedisKey.NOT_RELATION_EN_ANSWER.value.format(self.group_id))[
                            'prompt_content']
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=unrelated_question_answer_en,
                        only_assistant=unrelated_question_answer_en
                    )
                    return {
                        "data": unrelated_question_answer_en, "doc": [],
                        "generate_new_question_prompt_content": generate_new_question_prompt_content,
                        "generate_new_question": self.new_question,
                        "history_messages": history_message,
                        "chat_id": self.chat_id,
                        'duration':
                            {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                    }

        # 查询相关文档
        vector_store = common_dependencies()
        search_documents_start_time = get_shanghai_timestamp_ms()
        logger.info(f'搜索文档的开始时间:::{timestamp_to_datetime(search_documents_start_time)}')
        search_documents = vector_store['documents_vector_store'].similarity_search_with_score(query=self.new_question,
                                                                                               k=self.doc_count,
                                                                                               search_type="script_scoring",
                                                                                               space_type=self.space_type)
        search_documents_end_time = get_shanghai_timestamp_ms()
        logger.info(f'搜索文档的结束时间:::{timestamp_to_datetime(search_documents_end_time)}')
        self.search_doc_time = search_documents_end_time - search_documents_start_time

        documents = [x for (x, y) in search_documents if y >= self.search_threshold]

        # 无文档逻辑
        if len(documents) == 0:
            if '中文' in self.language:
                if self.chat_model == 'system':
                    # 这里查询中文、无文档、system的prompt
                    cn_no_doc_system_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_NO_DOC_CN_SYSTEM_PROMPT.value.format(self.group_id))

                    cn_no_doc_system_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                                  prompt=cn_no_doc_system_chat_prompt[
                                                                                      'prompt_content'],
                                                                                  parameter=
                                                                                  cn_no_doc_system_chat_prompt[
                                                                                      'prompt_params'].split(','))
                    cn_no_doc_system_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文没有文档question在system的开始时间:::{timestamp_to_datetime(cn_no_doc_system_chat_start_time)}')
                    cn_no_doc_system_chat_date = cn_no_doc_system_chat.run(
                        {'question': self.new_question, "history_message": history_message})
                    cn_no_doc_system_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文没有文档question在system的结束时间:::{timestamp_to_datetime(cn_no_doc_system_chat_end_time)}')
                    self.final_chain_time = cn_no_doc_system_chat_end_time - cn_no_doc_system_chat_start_time

                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time

                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=cn_no_doc_system_chat_date,
                        only_assistant=cn_no_doc_system_chat_date
                    )

                    return {"data": cn_no_doc_system_chat_date, "doc": [],
                            "prompt": cn_no_doc_system_chat_prompt['prompt_content'],
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "generate_new_question": self.new_question,
                            "history_messages": history_message,
                            "chat_id": self.chat_id,
                            'duration': {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                            }
                else:
                    # 这里查询中文、无文档、human的prompt
                    cn_no_doc_human_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_NO_DOC_CN_HUMAN_PROMPT.value.format(self.group_id))
                    messages = ["SYSTEM:" + cn_no_doc_human_chat_prompt['prompt_content'], "HUMAN:{question}"]
                    cn_no_doc_human_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                                 prompt="\n".join(messages),
                                                                                 parameter=cn_no_doc_human_chat_prompt[
                                                                                               'prompt_params'].split(
                                                                                     ',') + ["question"])
                    cn_no_doc_human_chat_date_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文没有文档question在human的开始时间:::{timestamp_to_datetime(cn_no_doc_human_chat_date_start_time)}')
                    cn_no_doc_human_chat_date = cn_no_doc_human_chat.run(
                        {'question': self.new_question, "history_message": history_message})
                    cn_no_doc_human_chat_date_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文没有文档question在human的结束时间:::{timestamp_to_datetime(cn_no_doc_human_chat_date_end_time)}')
                    self.final_chain_time = cn_no_doc_human_chat_date_end_time - cn_no_doc_human_chat_date_start_time

                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=cn_no_doc_human_chat_date,
                        only_assistant=cn_no_doc_human_chat_date
                    )
                    return {"data": cn_no_doc_human_chat_date, "doc": [],
                            "prompt": "\n".join(messages),
                            "generate_new_question": self.new_question,
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "history_messages": history_message,
                            "chat_id": self.chat_id,
                            'duration': {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                            }
            else:
                if self.chat_model == 'system':
                    # 这里查询英文、无文档、system的prompt
                    en_no_doc_system_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_NO_DOC_EN_SYSTEM_PROMPT.value.format(self.group_id))
                    en_no_doc_system_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                                  prompt=en_no_doc_system_chat_prompt[
                                                                                      'prompt_content'],
                                                                                  parameter=
                                                                                  en_no_doc_system_chat_prompt[
                                                                                      'prompt_params'].split(','))
                    en_no_doc_system_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文没有文档question在system的开始时间:::{timestamp_to_datetime(en_no_doc_system_chat_start_time)}')
                    en_no_doc_system_chat_date = en_no_doc_system_chat.run(
                        {'question': self.new_question, "history_message": history_message})
                    en_no_doc_system_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文没有文档question在system的结束时间:::{timestamp_to_datetime(en_no_doc_system_chat_end_time)}')
                    self.final_chain_time = en_no_doc_system_chat_end_time - en_no_doc_system_chat_start_time
                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=en_no_doc_system_chat_date,
                        only_assistant=en_no_doc_system_chat_date
                    )
                    return {"data": en_no_doc_system_chat_date, "doc": [],
                            "prompt": en_no_doc_system_chat_prompt['prompt_content'],
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "generate_new_question": self.new_question,
                            "history_messages": history_message,
                            "chat_id": self.chat_id,
                            'duration': {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                            }
                else:
                    # 这里查询英文、无文档、human的prompt
                    en_no_doc_human_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_NO_DOC_EN_HUMAN_PROMPT.value.format(self.group_id))
                    messages = ["SYSTEM:" + en_no_doc_human_chat_prompt['prompt_content'], "HUMAN:{question}"]

                    en_no_doc_human_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                                 prompt="\n".join(messages),
                                                                                 parameter=en_no_doc_human_chat_prompt[
                                                                                               'prompt_params'].split(
                                                                                     ',') + ["question"])
                    en_no_doc_human_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文没有文档question在human的开始时间:::{timestamp_to_datetime(en_no_doc_human_chat_start_time)}')
                    en_no_doc_human_chat_data = en_no_doc_human_chat.run(
                        {'question': self.new_question, "history_message": history_message})
                    en_no_doc_human_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文没有文档question在human结束时间:::{timestamp_to_datetime(en_no_doc_human_chat_end_time)}')

                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time

                    self.final_chain_time = en_no_doc_human_chat_end_time - en_no_doc_human_chat_start_time
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=en_no_doc_human_chat_data,
                        only_assistant=en_no_doc_human_chat_data
                    )
                    return {"data": en_no_doc_human_chat_data, "doc": [],
                            "prompt": "\n".join(messages),
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "generate_new_question": self.new_question,
                            "history_messages": history_message,
                            "chat_id": self.chat_id,
                            'duration': {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                            }
        # 有文档逻辑
        else:
            documents_en = [x.page_content for x in documents]
            final_documents = documents_en
            documents_cn = ''
            if '中文' in self.language:
                if self.is_translate_doc == 1:
                    # 先调用翻译，把文档片段翻译为中文
                    translate_prompt_obj = RedisService.get_prompt(
                        RedisKey.QUESTION_TRANSLATE_PROMPT.value.format(self.group_id))
                    translate_chain = AllPurposeChain.universal_llm_adapter(llm=self.common_glm_llm,
                                                                            prompt=translate_prompt_obj[
                                                                                'prompt_content'],
                                                                            parameter=translate_prompt_obj[
                                                                                'prompt_params'].split(','))
                    translate_doc_start_time = get_shanghai_timestamp_ms()
                    logger.info(f'翻译搜索出来文档的开始时间:::{timestamp_to_datetime(translate_doc_start_time)}')

                    documents_cn = translate_chain.run(
                        {'content': ".".join([x.page_content for x in documents]), 'from_str': '英文',
                         'to_str': '简体中文'})
                    final_documents = documents_cn
                    translate_doc_end_time = get_shanghai_timestamp_ms()

                    logger.info(f'翻译搜索出来文档的结束时间:::{timestamp_to_datetime(translate_doc_end_time)}')

                    self.language_translate_doc_time = translate_doc_end_time - translate_doc_start_time

                if self.chat_model == 'system':
                    #  这里查询中文、有文档、system的prompt
                    cn_doc_system_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_DOC_CN_SYSTEM_PROMPT.value.format(self.group_id))
                    cn_doc_system_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                               prompt=cn_doc_system_chat_prompt[
                                                                                   'prompt_content'],
                                                                               parameter=cn_doc_system_chat_prompt[
                                                                                   'prompt_params'].split(','))
                    cn_doc_system_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文有文档question在system的开始时间:::{timestamp_to_datetime(cn_doc_system_chat_start_time)}')
                    cn_doc_system_chat_date = cn_doc_system_chat.run(
                        {'question': self.new_question, "context": final_documents, "history_message": history_message})
                    cn_doc_system_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文有文档question在system的结束时间:::{timestamp_to_datetime(cn_doc_system_chat_end_time)}')

                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time
                    self.final_chain_time = cn_doc_system_chat_end_time - cn_doc_system_chat_start_time
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=cn_doc_system_chat_date,
                        only_assistant=cn_doc_system_chat_date
                    )
                    return {"data": cn_doc_system_chat_date,
                            "doc_en": documents_en,
                            "doc_cn": documents_cn,
                            "prompt": cn_doc_system_chat_prompt['prompt_content'],
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "generate_new_question": self.new_question,
                            "history_messages": history_message,
                            "chat_id": self.chat_id,
                            'duration': {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                            }
                else:
                    #  这里查询中文、有文档、human的prompt
                    cn_doc_human_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_DOC_CN_HUMAN_PROMPT.value.format(self.group_id))
                    messages = ["SYSTEM:" + cn_doc_human_chat_prompt['prompt_content'], "HUMAN:{question}"]
                    cn_doc_human_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                              prompt="\n".join(messages),
                                                                              parameter=cn_doc_human_chat_prompt[
                                                                                            'prompt_params'].split(
                                                                                  ',') + ["question"])
                    cn_doc_human_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文有文档question在human的开始时间:::{timestamp_to_datetime(cn_doc_human_chat_start_time)}')
                    cn_doc_human_chat_data = cn_doc_human_chat.run(
                        {'question': self.new_question, "context": final_documents, "history_message": history_message})
                    cn_doc_human_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'中文有文档question在human的结束时间:::{timestamp_to_datetime(cn_doc_human_chat_end_time)}')

                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time
                    self.final_chain_time = cn_doc_human_chat_end_time - cn_doc_human_chat_start_time

                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=cn_doc_human_chat_data,
                        only_assistant=cn_doc_human_chat_data
                    )
                    return {"data": cn_doc_human_chat_data,
                            "doc_en": documents_en,
                            "doc_cn": documents_cn,
                            "prompt": "\n".join(messages),
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "generate_new_question": self.new_question,
                            "history_messages": history_message,
                            "chat_id": self.chat_id,
                            'duration': {
                                "total_time": self.total_time,
                                "language_estimate_time": self.language_estimate_time,
                                "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                                "generate_new_question_time": self.generate_new_question_time,
                                "language_translate_question_time": self.language_translate_question_time,
                                "search_doc_time": self.search_doc_time,
                                "language_translate_doc_time": self.language_translate_doc_time,
                                "final_chain_time": self.final_chain_time,
                            }
                            }
            else:
                if self.chat_model == 'system':
                    #  这里查询英文、有文档、system的prompt
                    en_doc_system_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_DOC_EN_SYSTEM_PROMPT.value.format(self.group_id))
                    en_doc_system_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                               prompt=en_doc_system_chat_prompt[
                                                                                   'prompt_content'],
                                                                               parameter=en_doc_system_chat_prompt[
                                                                                   'prompt_params'].split(','))
                    en_doc_system_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文有文档question在system的开始时间:::{timestamp_to_datetime(en_doc_system_chat_start_time)}')
                    en_doc_system_chat_data = en_doc_system_chat.run(
                        {'question': self.new_question, "context": ".".join(documents_en),
                         "history_message": history_message})
                    en_doc_system_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文有文档question在system的结束时间:::{timestamp_to_datetime(en_doc_system_chat_end_time)}')
                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time
                    self.final_chain_time = en_doc_system_chat_end_time - en_doc_system_chat_start_time
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=en_doc_system_chat_data,
                        only_assistant=en_doc_system_chat_data
                    )
                    return {
                        "data": en_doc_system_chat_data,
                        "doc_en": documents_en, "prompt": en_doc_system_chat_prompt['prompt_content'],
                        "generate_new_question": self.new_question,
                        "generate_new_question_prompt_content": generate_new_question_prompt_content,
                        "history_messages": history_message,
                        "chat_id": self.chat_id,
                        'duration': {
                            "total_time": self.total_time,
                            "language_estimate_time": self.language_estimate_time,
                            "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                            "generate_new_question_time": self.generate_new_question_time,
                            "language_translate_question_time": self.language_translate_question_time,
                            "search_doc_time": self.search_doc_time,
                            "language_translate_doc_time": self.language_translate_doc_time,
                            "final_chain_time": self.final_chain_time,
                        }
                    }
                else:
                    #  这里查询英文、有文档、human的prompt
                    en_doc_human_chat_prompt = RedisService.get_prompt(
                        RedisKey.RESULT_DOC_EN_HUMAN_PROMPT.value.format(self.group_id))
                    messages = ["SYSTEM:" + en_doc_human_chat_prompt['prompt_content'], "HUMAN:{question}"]
                    en_doc_human_chat = AllPurposeChain.universal_llm_adapter(llm=self.chat_glm_llm,
                                                                              prompt="\n".join(messages),
                                                                              parameter=en_doc_human_chat_prompt[
                                                                                            'prompt_params'].split(
                                                                                  ',') + ["question"])
                    en_doc_human_chat_start_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文有文档question在human的开始时间:::{timestamp_to_datetime(en_doc_human_chat_start_time)}')
                    en_doc_human_chat_data = en_doc_human_chat.run(
                        {'question': self.new_question, "context": ".".join(documents_en),
                         "history_message": history_message})
                    en_doc_human_chat_end_time = get_shanghai_timestamp_ms()
                    logger.info(
                        f'英文有文档question在human的结束时间:::{timestamp_to_datetime(en_doc_human_chat_end_time)}')

                    requests_end_time = get_shanghai_timestamp_ms()
                    logger.info(f'请求的结束时间:::{timestamp_to_datetime(requests_end_time)}')
                    self.total_time = requests_end_time - requests_start_time
                    self.final_chain_time = en_doc_human_chat_end_time - en_doc_human_chat_start_time
                    update_message_by_id(
                        chat_id=self.chat_id,
                        message_id=streamed_chat_history.message_id,
                        user_message=self.question,
                        assistant=en_doc_human_chat_data,
                        only_assistant=en_doc_human_chat_data
                    )
                    return {
                        "data": en_doc_human_chat_data,
                        "doc_en": documents_en, "prompt": "\n".join(messages),
                        "generate_new_question": self.new_question,
                        "history_messages": history_message,
                        "chat_id": self.chat_id,
                        'duration': {
                            "total_time": self.total_time,
                            "generate_new_question_prompt_content": generate_new_question_prompt_content,
                            "language_estimate_time": self.language_estimate_time,
                            "estimate_semiconductor_question_time": self.estimate_semiconductor_question_time,
                            "generate_new_question_time": self.generate_new_question_time,
                            "language_translate_question_time": self.language_translate_question_time,
                            "search_doc_time": self.search_doc_time,
                            "language_translate_doc_time": self.language_translate_doc_time,
                            "final_chain_time": self.final_chain_time,
                        }
                    }
