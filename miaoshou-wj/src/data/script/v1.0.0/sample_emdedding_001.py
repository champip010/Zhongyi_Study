import json
import os
import time
import uuid

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch


def text_vector(text: str, file_name: str, file_path: str, paper_title):
    text_splitter = CharacterTextSplitter(
        separator="，",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    docs = text_splitter.create_documents([text])
    print(f'文件{file_path}切分{len(docs)}个chunk')
    documents_vector_store = get_documents_vector_store()
    # time.sleep(30)

    vector_part_ids = []
    vector_ids = []
    doc_with_metadatas = []
    size = 0

    for doc in docs:
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "paper_title": paper_title
        }
        doc_with_metadatas.append(Document(page_content=doc.page_content, metadata=metadata))
        size += 1
        id = str(uuid.uuid4())
        vector_part_ids.append(id)
        vector_ids.append(id)
        if size >= 200:  # 达到bulk_size
            print(f'准备发送')
            time.sleep(10)
            documents_vector_store.add_documents(doc_with_metadatas, ids=vector_part_ids, bulk_size=1000)
            size = 0
            vector_part_ids = []
            doc_with_metadatas = []
        #doc_with_metadata = Document(page_content=doc.page_content, metadata=metadata)
        #documents_vector_store.add_documents([doc_with_metadata], ids=[uuid.uuid4()],bulk_size=1000)
    if size > 0:
        time.sleep(10)
        print(f'准备发送')
        documents_vector_store.add_documents(doc_with_metadatas, ids=vector_part_ids, bulk_size=1000)
    print('此文档完成')
def get_documents_vector_store():
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        openai_api_key='504e77668fd846bd9c637c61c5f748df',
        openai_api_type='azure',
        openai_api_base='https://zkrt-embedding-20240201.openai.azure.com/'

    )
    return OpenSearchVectorSearch(opensearch_url="http://10.100.10.20:9200",
                                  index_name="semiconductor_test2_cn_1_vectors",
                                  embedding_function=embeddings, is_aoss=False)






# if __name__ == '__main__':
#     text_vector(text='''四. 使用物理方法修复快坏掉的机械硬盘如果在尝试了上述的几个方法后，你的机械硬盘（HDD）依然无法被你的电脑或其他工作正常的电脑所识别，那么很可能你的机械硬盘就是属于机械故障了。
#     ''', file_name='测试', file_path='测试', paper_title='测试')





if __name__ == '__main__':
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    total_token = 0
    with open("arxiv1_complete_file_001.txt", "a+") as f:
        f.seek(0)
        x = f.readlines()
        complete_file = [k.replace("\n", "") for k in x]

        folder_path = '/Users/zhangzhiyong/Documents/orgXueweiTxt-plumber/001'
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for index, file_name in enumerate(files):
            print(f'正在Embedding第{index+1}份文件')
            file_path = os.path.join(folder_path, file_name)
            if file_path not in complete_file:
                with open(file_path, 'r') as file:
                    data = file.read()
                    # token = encoding.encode(data)
                    # total_token += len(token)
                    file_name = os.path.basename(file_path)
                    paper_title = file_name.split(".")[0]
                    text_vector(text=data, file_name=file_name, file_path='', paper_title=paper_title)
                f.write(file_path+"\n")
                print(file_name,'\n')

            else:
                print(f"{file_name}已经存在，跳过")

    print(f'一共Token{total_token}')