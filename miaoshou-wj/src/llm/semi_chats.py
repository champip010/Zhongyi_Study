import re

from langchain_community.chat_models import ChatOpenAI

from baby_agi.all_purpose_chain import create_increment_prompt_universal
from llm.ai_llm_utils import get_gpt4_client, get_llm_bean
from utils.redisutils import RedisService


# 有文档
#   human
#     1. 中文
#     2. 英文
#   chat
#     3. 中文
#     4. 英文
# 无文档
#   human
#     7. 中文
#     8. 英文
#   chat
#     5. 中文
#     6. 英文

class PromptNameConstant:
    DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_CN = 'doc_chat_answer_question_in_system_cn'
    DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_EN = 'doc_chat_answer_question_in_system_en'
    DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_CN = 'doc_chat_answer_question_in_human_cn'
    DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_EN = 'doc_chat_answer_question_in_human_en'
    NO_DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_CN = 'no_doc_chat_answer_question_in_system_cn'
    NO_DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_EN = 'no_doc_chat_answer_question_in_system_en'
    NO_DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_CN = 'no_doc_chat_answer_question_in_human_cn'
    NO_DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_EN = 'no_doc_chat_answer_question_in_human_en'
    UNRELATED_QUESTION_ANSWER_CN = 'unrelated_question_answer_cn'
    UNRELATED_QUESTION_ANSWER_EN = 'unrelated_question_answer_en'
    # #
    LANGUAGE_ESTIMATE = 'language_estimate'
    LANGUAGE_TRANSLATE = 'language_translate'
    CHAT_TITLE = 'chat_title'
    ESTIMATE_SEMICONDUCTOR_QUESTION = 'estimate_semiconductor_question'
    SIMILARITY_SEARCH_THRESHOLD_VALUE = 'similarity_search_threshold_value'
    SIMILARITY_SEARCH_ARITHMETIC = 'similarity_search_arithmetic'
    FINAL_ANSWER_LLM_TEMPERATURE = 'final_answer_llm_temperature'
    QUESTION_IN_WHERE: str = 'question_in_where'
    IS_TRANSLATE_DOC: str = 'is_translate_doc'
    QUERY_HISTORY_MESSAGE_COUNT = 'query_history_message_count'
    GENERATE_NEW_QUESTION = 'generate_new_question'
    QUERY_DOC_NUMBER = 'query_doc_number'
    IS_ESTIMATE_QUESTION = "is_estimate_question"
    ESTIMATE_SEMICONDUCTOR_HEALTHY_QUESTION = "estimate_semiconductor_healthy_question"
    EXTRACT_MERIDIAN_NAMES = "extract_meridian_names"
    EXTRACT_DISEASE_NAMES = "extract_disease_names"


def parse_prompt_and_prompt_params(prompt_name: str, prompt_group_id: str = None):
    if not prompt_group_id:
        prompt_group_id = RedisService.get_prompt_group_id()
    redis_prompt = RedisService.get_prompt("system:" + prompt_group_id + ":" + prompt_name)
    prompt_content = redis_prompt['prompt_content']
    prompt_params = redis_prompt['prompt_params']
    return prompt_content, prompt_params.split(",")


def language_translate_result(content: str, from_str: str, to_str: str, prompt_group_id: str, llm: ChatOpenAI):
    language_estimate_prompt, language_translate_params = parse_prompt_and_prompt_params(
        prompt_name=PromptNameConstant.LANGUAGE_TRANSLATE, prompt_group_id=prompt_group_id)

    language_translate_chain = create_increment_prompt_universal(llm,
                                                                 language_estimate_prompt,
                                                                 language_translate_params)

    language_translate_dict = {
        'content': content,
        'from_str': from_str,
        'to_str': to_str,
    }

    language_translate_response = language_translate_chain.run(
        language_translate_dict
    )
    return language_translate_response


def llm_estimate_question(question: str, prompt_name: str, prompt_group_id: str, history_message: list) -> bool:
    estimate_question_result_llm = get_gpt4_client()
    estimate_semiconductor_question_prompt, estimate_semiconductor_question_params = parse_prompt_and_prompt_params(
        prompt_name=prompt_name, prompt_group_id=prompt_group_id)
    estimate_semiconductor_question_chain = create_increment_prompt_universal(
        estimate_question_result_llm,
        estimate_semiconductor_question_prompt,
        estimate_semiconductor_question_params
    )
    language_translate_dict = {
        'question': question,
        'history_message': history_message,
    }
    estimate_semiconductor_question_response = estimate_semiconductor_question_chain.run(
        language_translate_dict
    )
    rlt = True
    if "No" in estimate_semiconductor_question_response or "no" in estimate_semiconductor_question_response:
        rlt = False
    return rlt


def estimate_semiconductor_question_result(question: str, prompt_group_id: str, history_message: list):
    estimate_question_result_llm = get_gpt4_client()
    estimate_semiconductor_question_prompt, estimate_semiconductor_question_params = parse_prompt_and_prompt_params(
        prompt_name=PromptNameConstant.ESTIMATE_SEMICONDUCTOR_QUESTION, prompt_group_id=prompt_group_id)

    estimate_semiconductor_question_chain = create_increment_prompt_universal(estimate_question_result_llm,
                                                                              estimate_semiconductor_question_prompt,
                                                                              estimate_semiconductor_question_params)

    language_translate_dict = {
        'question': question,
        'history_message': history_message,
    }

    estimate_semiconductor_question_response = estimate_semiconductor_question_chain.run(
        language_translate_dict
    )

    rlt = 1
    if "No" in estimate_semiconductor_question_response or "no" in estimate_semiconductor_question_response:
        rlt = 0

    return rlt


def generate_new_question(question: str, history_message: list, prompt_group_id: str):
    """

    @param question:  原始问题
    @param history_message:
    @return: 改写后的问题
    """
    llm = get_llm_bean(0)
    generate_new_question_prompt, generate_new_question_params = parse_prompt_and_prompt_params(
        prompt_name=PromptNameConstant.GENERATE_NEW_QUESTION, prompt_group_id=prompt_group_id)

    generate_new_question_params_chain = create_increment_prompt_universal(llm,
                                                                           generate_new_question_prompt,
                                                                           generate_new_question_params)
    generate_new_question_dict = {
        "message_history": history_message,
        "question": question
    }

    print(f"llm:{llm} generate_new_question_params_chain:{generate_new_question_params_chain} generate_new_question_dict:{generate_new_question_dict}")

    generate_new_question_resp = generate_new_question_params_chain.run(generate_new_question_dict)

    return generate_new_question_resp


async def get_text_en_cn_count(text):
    ##########################################################
    # 使用正则表达式匹配中文字符，中文编码范围一般为\u4e00到\u9fff
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 使用正则表达式匹配英文单词，假设单词由字母数字及下划线组成
    english_pattern = re.compile(r'\b\w+\b')

    # 统计中文字符数量
    chinese_characters = re.findall(chinese_pattern, text)
    chinese_count = len(chinese_characters)

    # 统计英文单词数量
    english_words = re.findall(english_pattern, text)
    english_count = sum([1 for word in english_words if re.match(r'^[A-Za-z]+$', word)])

    return chinese_count, english_count


# 判断text中英文，返回0中文，返回1英文
async def identify_language(text):
    chinese_count, english_count = await get_text_en_cn_count(text)
    # 判断哪种语言的字符数量更多
    chNum = 2 * chinese_count
    if chNum > english_count:
        return 0
    else:
        return 1


async def ai_polish_identify_language_en(text):
    chinese_count, english_count = await get_text_en_cn_count(text)
    # 计算英文字符占比
    english_ratio = english_count / (chinese_count + english_count)
    # 判断英文字符占比是否超过90%
    return english_ratio >= 0.9


async def ai_polish_identify_language_cn(text):
    chinese_count, english_count = await get_text_en_cn_count(text)
    # 计算中文字符占比
    chinese_ratio = chinese_count / (chinese_count + english_count)
    # 判断中文字符占比是否超过90%
    return chinese_ratio >= 0.9


async def ai_polish_identify_language(text):
    """
    判断text 是否包含两种语言，如果只有一种语言返回语言种类
    """
    chinese_count, english_count = await get_text_en_cn_count(text)
    # 计算中文字符占比
    chinese_ratio = chinese_count / (chinese_count + english_count)
    english_ratio = 1 - chinese_ratio
    # 判断中文字符占比是否超过90%
    if chinese_ratio >= 0.9:
        return True, "cn"
    elif english_ratio >= 0.9:
        return True, "en"
    else:
        return False, "unknown"


def get_final_chain(
        question_in_where: str,  # 有文档的链最后是否按照HumanMessagePromptTemplate  -- 暴露问题
        from_str: str,  # 用户的输入语言
        prompt_group_id: str,  # prompt组ID
        llm_stream: ChatOpenAI,
        llm: ChatOpenAI,
        is_exist_doc: bool,
        doc_list: list,
        question: str
):
    if is_exist_doc:
        # Human
        if 'human' in question_in_where.lower():
            if from_str == '简体中文':
                is_translate_doc_value, is_translate_doc_params = parse_prompt_and_prompt_params(
                    prompt_name=PromptNameConstant.IS_TRANSLATE_DOC, prompt_group_id=prompt_group_id)
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_CN, prompt_group_id)
                doc_source_text = ''
                for doc in doc_list:
                    doc_source_text += (doc.page_content + '.')
                if is_translate_doc_value == '1':
                    doc_context = language_translate_result(content=doc_source_text, from_str='英文',
                                                            to_str='简体中文', prompt_group_id=prompt_group_id,
                                                            llm=llm)
                else:
                    doc_context = doc_source_text

            else:
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_EN,
                    prompt_group_id)
                doc_context = ''
                for doc in doc_list:
                    doc_context += (doc.page_content + '.')

            prompt_messages = ["SYSTEM:" + prompt_content, "HUMAN:{question}"]
            prompt_params.append('question')
            final_chain = create_increment_prompt_universal(llm_stream,
                                                            "\n".join(prompt_messages),
                                                            prompt_params)

            final_chain_dict = {
                'context': doc_context,
                'question': question
            }

        # LLMChain方式
        else:
            if from_str == '简体中文':
                is_translate_doc_value, is_translate_doc_params = parse_prompt_and_prompt_params(
                    prompt_name=PromptNameConstant.IS_TRANSLATE_DOC, prompt_group_id=prompt_group_id)
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_CN,
                    prompt_group_id)
                doc_source_text = ''
                for doc in doc_list:
                    doc_source_text += (doc.page_content + '.')
                # 是否翻译文档
                if is_translate_doc_value == '1':
                    doc_context = language_translate_result(content=doc_source_text, from_str='英文',
                                                            to_str='简体中文', prompt_group_id=prompt_group_id,
                                                            llm=llm)
                else:
                    doc_context = doc_source_text
            else:
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_EN,
                    prompt_group_id)
                doc_context = ''
                for doc in doc_list:
                    doc_context += (doc.page_content + '.')

            final_chain = create_increment_prompt_universal(llm_stream,
                                                            prompt_content,
                                                            prompt_params)
            final_chain_dict = {
                'context': doc_context,
                'question': question
            }
    else:
        # Human
        if 'human' in question_in_where.lower():
            if from_str == '简体中文':
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.NO_DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_CN, prompt_group_id)
            else:
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.NO_DOC_CHAT_ANSWER_QUESTION_IN_HUMAN_EN,
                    prompt_group_id)
            prompt_messages = ["SYSTEM:" + prompt_content, "HUMAN:{question}"]

            prompt_params.append("question")
            final_chain = create_increment_prompt_universal(llm_stream,
                                                            "\n".join(prompt_messages),
                                                            prompt_params)

            final_chain_dict = {
                'question': question
            }

        # LLMChain方式
        else:
            if from_str == '简体中文':
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.NO_DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_CN, prompt_group_id)
            else:
                prompt_content, prompt_params = parse_prompt_and_prompt_params(
                    PromptNameConstant.NO_DOC_CHAT_ANSWER_QUESTION_IN_SYSTEM_EN,
                    prompt_group_id)
            final_chain = create_increment_prompt_universal(llm_stream,
                                                            prompt_content,
                                                            prompt_params)
            final_chain_dict = {
                'question': question
            }

    return final_chain, final_chain_dict
