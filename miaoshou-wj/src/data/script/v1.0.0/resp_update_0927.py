import json

import requests


unrelated_question_answer_cn_prompt = "对不起，您的问题与半导体或大健康无关，请询问与之相关的"

unrelated_question_answer_en_prompt = "Sorry, your question is not related to semiconductors or big health. Please ask questions that are relevant to them."

unrelated_question_answer_cn_prompt_choice = {
    "prompt_name": "unrelated_question_answer_cn",
    "prompt_content": unrelated_question_answer_cn_prompt,
    "prompt_params": ""
}

unrelated_question_answer_en_prompt_choice = {
    "prompt_name": "unrelated_question_answer_en",
    "prompt_content": unrelated_question_answer_en_prompt,
    "prompt_params": ""
}

estimate_semiconductor_question_prompt_content = '''
## 角色
你是一个半导体领域问题过滤器，你的名字叫SemiGPT。
##目标 
根据聊天历史，来判断当前问题是否和物理或者化学或者半导体的科学、工程、技术、材料、设计、制造、测试或应用等领域是否相关。
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


estimate_semiconductor_question_prompt_choice = {
    "prompt_name": "estimate_semiconductor_question",
    "prompt_content": estimate_semiconductor_question_prompt_content,
    "prompt_params": ""
}


# 目标URL
url = 'https://semiai.semi.ac.cn/api/update/system/cache'
# url = 'https://test.scientific.ratubrain.com/api/update/system/cache'
# url = 'http://ratubrain-chatglm3-test.semiconductor-test.10.100.10.17.sslip.io/api/update/system/cache'
# url = 'https://paper.ratubrain.com/api/update/system/cache'
# url = 'http://192.168.110.90:5005/update/system/cache'
# url = 'http://paper-ratubrain.scientific-test.10.100.10.17.sslip.io/api/update/system/cache'
# 设置请求头
headers = {
    'Content-Type': 'application/json'
}

# 发送POST请求
response = requests.post(url, headers=headers, json=unrelated_question_answer_cn_prompt_choice)
# 打印响应内容
print('Response Status Code:', response.status_code)
print('Response Body:', response.text)

# response = requests.post(url, headers=headers, json=unrelated_question_answer_en_prompt_choice)
# # 打印响应内容
# print('Response Status Code:', response.status_code)
# print('Response Body:', response.text)
