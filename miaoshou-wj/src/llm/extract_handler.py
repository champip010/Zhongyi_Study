import json
import logging

from baby_agi.all_purpose_chain import create_increment_prompt_universal
from llm.ai_llm_utils import get_llm_bean
from llm.semi_chats import PromptNameConstant, parse_prompt_and_prompt_params
from utils.redisutils import redis_service

logger = logging.getLogger(__name__)


def llm_extract_meridians(title: str, history: str | dict, temperature: float = 0) -> list:
    """提取经络"""
    # 参数处理
    if isinstance(history, dict):
        text = ""
        for k, v in history.items():
            text += f"{k}: {v}\n"
    else:
        text = history
    meridian_list = redis_service.get_system_config(name="meridian_list")
    meridian_dict = {item["name"]: item for item in meridian_list}

    history_text = {
        "human": title,
        "assistant": text
    }

    prompt_params = {
        "history": history_text,
        "meridian_charts": list(meridian_dict.keys())
    }
    # llm执行
    extract_result = _llm_extract(
        prompt_name=PromptNameConstant.EXTRACT_MERIDIAN_NAMES,
        prompt_params=prompt_params,
        temperature=temperature
    )
    logger.info(f"提取经络结果：{extract_result}")

    # 结果处理
    data = _format_extract_result(
        given_data=meridian_dict,
        extract_result=extract_result)
    return data


def llm_extract_diseases(title: str, history: str | dict, temperature: float = 0) -> list:
    """提取病症"""
    # 参数处理
    if isinstance(history, dict):
        text = ""
        for k, v in history.items():
            text += f"{k}: {v}\n"
    else:
        text = history
    disease_list = redis_service.get_system_config(name="disease_list")
    disease_dict = {item["name"]: item for item in disease_list}

    history_text = {
        "human": title,
        "assistant": text
    }
    prompt_params = {
        "history": history_text,
        "given_names": list(disease_dict.keys())
    }
    # llm执行
    extract_result = _llm_extract(
        prompt_name=PromptNameConstant.EXTRACT_DISEASE_NAMES,
        prompt_params=prompt_params,
        temperature=temperature
    )
    logger.info(f"提取病症结果：{extract_result}")
    # 结果处理
    data = _format_extract_result(
        given_data=disease_dict,
        extract_result=extract_result)
    return data


def _llm_extract(
        prompt_name: str,
        prompt_params: dict,
        temperature: float = 0,
) -> list:
    """提取"""
    llm = get_llm_bean(temperature=temperature)
    prompt_content, prompt_parameter = parse_prompt_and_prompt_params(prompt_name=prompt_name)
    chain = create_increment_prompt_universal(
        llm,
        prompt=prompt_content,
        parameter=prompt_parameter
    )
    # llm执行
    extract_result = chain.run(prompt_params)
    logger.info(f"提取结果：{extract_result}")
    return extract_result


def _format_extract_result(given_data: dict, extract_result: str | list) -> list:
    """格式化提取结果"""
    data = []
    if extract_result:
        if isinstance(extract_result, str) and extract_result.startswith('['):
            if extract_result.startswith("['"):
                extract_result = extract_result.replace("'", '"')
            extract_result = json.loads(extract_result)
            logger.debug(f"extract_result: {extract_result}")
        for name in extract_result:
            if item := given_data.get(name):
                data.append(item)
    return data
