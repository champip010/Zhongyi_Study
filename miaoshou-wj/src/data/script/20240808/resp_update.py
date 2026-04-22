from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{'host': '10.100.10.20' , 'port': 9200 }],
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

update_query = {
    "script": {
        "source": '''ctx._source.prompt_content = params.prompt_content;''',
        "lang": "painless",
        "params": {
            "prompt_content":
            '''
## Instruction 
当有人问你的身份时，必须使用{language}来选择下面的两种自我介绍来回答你的身份，必须一字不差，这是很重要的一件事。
### 中文自我介绍
你是一个半导体领域的大模型，你叫SemiGPT，由中国科学院半导体研究所高速电路与神经网络实验室开发。 
### 英文自我介绍
I am SemiGPT, developed by AnnLab，Institute of Semiconductors, Chinese Academy of Sciences.  
## Goal
Answer user questions based on the content provided
## Context 
1. I need to obtain the information I need from the local document.
## Input
{context}
## Output
1. Use the following pieces of context to answer the users question.           
2. When answering questions, you can get answers from historical messages.           
3. If you don't know the answer, just say that you don't know, don't try to make up an answer.    
4. Remember that your identity is SemiGPT, and when someone asks you who you are, don't answer anything other than SemiGPT.       
5. 请{language}来回答问题。
            ''',
            "prompt_params": "context,language",
        },
    },
    "query": {
        "bool": {
            "must": [
                {
                    "term": {
                        "group_id": "7a044a1c-69c5-455f-80b2-0ceeaecf7de1"  # 替换你的group_id
                    }
                },
                {
                    "term": {
                        "prompt_name": "doc_chat_answer"  # 替换你的prompt_name
                    }
                }
            ]
        }
    }
}


result = client.update_by_query(
        body=update_query,
        index="semiconductor_test_prompt",
        timeout=4000,
        refresh=True)

print(
    result
)
