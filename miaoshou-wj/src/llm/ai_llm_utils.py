from langchain_community.chat_models import AzureChatOpenAI, ChatOpenAI
from openai import AsyncOpenAI

from models.settings import BrainSettings


def get_gpt4_client():
    estimate_question_result_llm = AzureChatOpenAI(temperature=0,
                                                   deployment_name=BrainSettings().azure_deployment_name,
                                                   openai_api_base=BrainSettings().azure_openai_api_base,
                                                   openai_api_key=BrainSettings().openai_api_key,
                                                   openai_api_version=BrainSettings().azure_openai_api_version
                                                   )
    return estimate_question_result_llm


def get_llm_bean(temperature: float):
    print(f"""model_name:{BrainSettings().glm4_model_name}  temperature:{temperature}
      key:{BrainSettings().glm4_openai_api_key} openai_api_base: {BrainSettings().glm4_openai_api_base} """)
    llm = ChatOpenAI(model_name=BrainSettings().glm4_model_name,
                     temperature=temperature,
                     openai_api_key=BrainSettings().glm4_openai_api_key,
                     openai_api_base=BrainSettings().glm4_openai_api_base
                     )
    return llm


def get_openai_llm_bean_async():
    client_llm = AsyncOpenAI(
        api_key=BrainSettings().glm4_openai_api_key,
        base_url=BrainSettings().glm4_openai_api_base
    )
    return client_llm,BrainSettings().glm4_model_name
