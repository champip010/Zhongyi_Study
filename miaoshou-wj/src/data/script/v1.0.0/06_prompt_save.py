import uuid

from opensearchpy import OpenSearch

from utils.dateutils import get_shanghai_timestamp_ms

# test
# chat_group_id = "93be5514-f23a-496e-b72d-13b623f411ac"
# prod
chat_group_id = "dcceed1e-1c16-40a2-9048-3b5f05544e2e"

unrelated_question_answer_cn_prompt = '''
对不起，您的问题与半导体或大健康无关，请询问与之相关的
'''
unrelated_question_answer_en_prompt = '''
Sorry, your question is not related to semiconductors or big health. Please ask questions that are relevant to them.
'''

estimate_semiconductor_healthy_questionprompt = '''
## 角色
你是一个健康医学领域问题过滤器，你的名字叫SemiGPT。
##目标 
根据聊天历史，来判断当前问题是否和健康、医疗、中医、痛风、失眠、高血压、糖尿病、高血脂等领域是否相关。
## 聊天历史
{history_message}
## 当前问题
{question}
## 
## 输出限制
1. 如果问题与身份认定或者问候有关时请回答Yes，
2. 如果相关，回答Yes，不要解释，不要输出其他内容；
3. 如果不相关，回答No，不要解释，不要输出其他内容。
'''


estimate_semiconductor_healthy_cn_prompt = '''
你是一个医学专家，请根据如下信息和聊天历史针对下面的问题使用中文回答问题。如果患者的问题是关于疾病诊断的，则从食疗、运动、手法(针灸、穴位按摩、经络推拿、穴位热灸等)、物理刺激(力电声热光磁等)疗法、医院就诊等五个方面给出建议。如果患者问的是其他医学问题则直接回答
# 限制
1. 我提供参考文档中含有的内容，必须把内容全部输出，且不能做太多修改。
2. 输出的内容必须条理清晰且符合逻辑。
3. 无论上下文说什么，回答必须是中文。
4. 输出格式必须是markdown格式。

## 参考文档
{context}
##问题：
{question}
##聊天历史：
{history_message}       
'''

estimate_semiconductor_healthy_en_prompt = ''' 
You are a medical expert. Please answer the following questions in Chinese based on the following information and chat history. If the patient's question is about disease diagnosis, then give suggestions from five aspects: diet, exercise, manipulation (acupuncture and moxibustion, acupoint massage, meridian massage, acupoint thermal moxibustion, etc.), physical stimulation (force, electricity, sound, heat, light, magnetism, etc.), and hospital treatment. If the patient asks other medical questions, answer directly
#Restrictions
1. I provide the content contained in the reference document, and I must output all the content without making too many modifications.
2. The output content must be clear and logical.
3. No matter what the context says, the answer must be in Chinese.
4. The output format must be in markdown format.

## Reference Document 
{context} 
##Question:
{question}
##Chat history: 
{history_message}
'''

chat_prompt = [
    # {
    #     "prompt_name": "unrelated_question_answer_cn",
    #     "prompt_content": unrelated_question_answer_cn_prompt,
    #     "prompt_params": "",
    #     "group_id": chat_group_id,
    #     "create_time": get_shanghai_timestamp_ms(),
    #     "update_time": get_shanghai_timestamp_ms(),
    #     "order": 13,
    #     "name": "不是半导体问题的中文提示语"
    # },
    # {
    #     "prompt_name": "unrelated_question_answer_en",
    #     "prompt_content": unrelated_question_answer_en_prompt,
    #     "prompt_params": "",
    #     "group_id": chat_group_id,
    #     "create_time": get_shanghai_timestamp_ms(),
    #     "update_time": get_shanghai_timestamp_ms(),
    #     "order": 14,
    #     "name": "不是半导体问题的英文提示语"
    # },
    {
        "prompt_name": "estimate_semiconductor_healthy_question",
        "prompt_content": estimate_semiconductor_healthy_questionprompt,
        "prompt_params": "question,message_history",
        "group_id": chat_group_id,
        "create_time": get_shanghai_timestamp_ms(),
        "update_time": get_shanghai_timestamp_ms(),
        "order": 41,
        "name": "判断问题是否与数字健康相关"
    },
    {
        "prompt_name": "estimate_semiconductor_healthy_cn",
        "prompt_content": estimate_semiconductor_healthy_cn_prompt,
        "prompt_params": "question,history_message",
        "group_id": chat_group_id,
        "create_time": get_shanghai_timestamp_ms(),
        "update_time": get_shanghai_timestamp_ms(),
        "order": 42,
        "name": "数字健康相关问题，中文问题"
    },
    {
        "prompt_name": "estimate_semiconductor_healthy_en",
        "prompt_content": estimate_semiconductor_healthy_en_prompt,
        "prompt_params": "question,history_message",
        "group_id": chat_group_id,
        "create_time": get_shanghai_timestamp_ms(),
        "update_time": get_shanghai_timestamp_ms(),
        "order": 43,
        "name": "数字健康相关问题，英文问题"
    }
]

client = OpenSearch(
    # prod
    # hosts=[{'host': '172.16.0.165', 'port': 9200}],
    hosts=[{'host': '10.100.10.16', 'port': 9200}],
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

for prompt in chat_prompt:
    id = uuid.uuid4()
    client.index(
        index="semiconductor_test_prompt",
        body=prompt,
        id=str(id),
        refresh=True
    )
