import json
import os
import random
import time
import uuid
from typing import List

import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
# DEV
BCE_URL_LIST = 'http://172.16.44.174:9999/list/encode'
BCE_URL = 'http://172.16.44.174:9999/encode'

opensearch_list = ['http://172.16.44.30:17200', 'http://172.16.44.30:18200', 'http://172.16.44.30:19200',
                   'http://172.16.44.157:17200', 'http://172.16.44.157:18200', 'http://172.16.44.157:19200',
                   'http://172.16.0.165:9200', 'http://172.16.0.168:9200', 'http://172.16.0.167:9200']
INDEX_NAME = 'semiconductor_prod_vectors_new'


class CusBceEmbedding(Embeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
      #  time.sleep(2)
        models_body = {
            "input": texts
        }
        headers = {'Content-Type': 'application/json'}
        #d = requests.post(url=BCE_URL_LIST, json=models_body, headers=headers)
        result = requests.post(url=BCE_URL_LIST, json=models_body, headers=headers).json()

        return result['data']

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

        models_body = {
            "input": text
        }
        headers = {'Content-Type': 'application/json'}
        result = requests.post(url=BCE_URL, json=models_body, headers=headers).json()

        return result['data'][0]

def parse_latex(data):
    title = data['title']
    keywords = data['keywords']
    authors = data['authors']
    abstract = ""
    strAllContent = title + "\n" + ",".join(keywords) + "\n" + ",".join(authors)

    paragraphs = data['paragraphs']
    for paragraph in paragraphs:
        # 摘要
        if 'abstract' == paragraph['type']:
            abstract = paragraph['paragraph']
            strAllContent = strAllContent + "\n" + abstract

        # 主体内容
        if 'body_div' == paragraph['type'] or 'back' == paragraph['type']:
            strAllContent = strAllContent + "\n" + paragraph['paragraph']

        # if 'body_figure_div' == paragraph['type']:
        #     print("实例图片：" + paragraph['paragraph'] + paragraph['path'] if 'path' in paragraph else '')

    return title, keywords, authors, abstract,  strAllContent


def text_vector(text: str, file_name: str, file_path: str, paper_title: str, abstract: str, authors, keywords):
    text_splitter = CharacterTextSplitter(
        separator=",",
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    docs = text_splitter.create_documents([text])
    print(f'文件{file_path}切分{len(docs)}个 chunk')
    documents_vector_store = get_documents_vector_store()
    count = 0
    data_list = []
    ids = []
    for doc in docs:
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "paper_title": paper_title,
            "abstract": abstract,
            "authors": authors,
            "keywords": keywords
        }
        doc_with_metadata = Document(page_content=doc.page_content, metadata=metadata)
        for i in range(1, 9):
            if count > 5:
                break
            # data_list.append(doc_with_metadata)
            try:
                documents_vector_store.add_documents([doc_with_metadata])
                count = 10
            except Exception as c:
                print(c)
                documents_vector_store = get_documents_vector_store()
                count += 1
                time.sleep(10)

        # ids.append(uuid.uuid4())

    print("data_list" + str(len(data_list)))



def get_documents_vector_store():
    i = random.randint(0, 8)
    embeddings = CusBceEmbedding()

    return OpenSearchVectorSearch(opensearch_url=opensearch_list[i],
                                  index_name=INDEX_NAME,
                                  embedding_function=embeddings, is_aoss=False)


# 批量读取方法
def find_files(directory, extension):
    pathList = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.' + extension):
                # print(os.path.join(root, file))
                pathList.append(os.path.join(root, file))
    return pathList


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except IOError:
        print("读取文件错误")
        return {}


if __name__ == '__main__':
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_token = 0
    folder_path = "G://20万文献数据_json//output_3"

    total_token = 0
    with open("../20241022/arxiv1_complete_file_output3.txt", "a+") as f:
        f.seek(0)
        x = f.readlines()
        complete_file = [k.replace("\n", "") for k in x]

        # folder_path = '/Users/zhangzhiyong/Documents/orgXueweiTxt-plumber/001'
        # pathList = [f for f in os.listdir(folder_path, "json") if os.path.isfile(os.path.join(folder_path, f))]
        pathList = find_files(folder_path, "json")
    # for file_path in pathList:
        for index, file_path in enumerate(pathList):

            print(f'正在 Embedding 第{index + 1} - {file_path} 份文件')
            if file_path in complete_file:
                print('文件已经执行过，跳过操作')
                continue

            data = read_json_file(file_path)
            if 'title' in data:
            # 读取 标题、摘要、全部内容拼接
                title, keywords, authors, abstract,  strAllContent = parse_latex(data)

            # token = encoding.encode(data)
            # total_token += len(token)
                text_vector(text=strAllContent,
                            file_name=title+".pdf",
                            file_path="relative_path",
                            paper_title=title,
                            abstract=abstract,
                            authors=authors,
                            keywords=keywords)
                f.write(file_path + "\n")
            # print(file_name, '\n')
            else:
                continue
            # print(file_name, '\n')

print(total_token)