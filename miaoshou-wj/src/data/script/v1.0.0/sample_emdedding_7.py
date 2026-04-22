import os
import re
import uuid

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document


def extract_sections(content: str):
    # 先使用正则表达式找到所有的section, subsection, 和 subsubsection
    pattern = r'\\(sub)*section\{([^\}]+)\}'
    sections = re.findall(pattern, content)

    # 使用字典来存储找到的部分和各自的索引
    section_content = {}

    # 为每一个section, subsection, subsubsection 创建一个包含其内容的条目
    for match in sections:
        section_type, section_title = match
        section_type = "section" if section_type == "" else section_type
        content_pattern = r'(\\' + section_type + r'\{' + re.escape(
            section_title) + r'\}(.*?)(' + r'\\(sub)*section|\\end\{document\}))'
        section_data = re.search(content_pattern, content, re.DOTALL)

        if section_data:
            # 保存section的内容，但不包括下一个section的开始标记
            section_text = section_data.group(2).strip()
            section_content[section_title] = section_text

    return section_content


def extract_titles_and_contents(s):
    results = []

    # 使用正则表达式以'{\\textit{' 开始，至下一个'{\\textit{' 或字符串结尾为结束进行匹配
    pattern = re.compile(r'{\\textit{([^}]+)}}(.*?)(?={\\textit{|$})', re.DOTALL)

    # 寻找所有匹配的部分
    matches = pattern.finditer(s)

    for match in matches:
        # 提取标题和对应的内容
        title = match.group(1).strip()
        content = match.group(2).strip()
        results.append((title, content))

    return results


def parse_latex(file_path):
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     content = file.read()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='iso-8859-1') as f:
            content = f.read()

    # 移除注释
    content_no_comments = re.sub(r'%.*?$', '', content, flags=re.MULTILINE)

    # 移除图片
    content_no_images = re.sub(r'\\begin{figure}.*?\\end{figure}', '', content_no_comments, flags=re.DOTALL)
    content_no_images = re.sub(r'\\begin{figure\*}.*?\\end{figure\*}', '', content_no_images, flags=re.DOTALL)
    # 去掉标签
    # content_no_images = re.sub(r'\\label.*?$', '', content_no_images)
    content_no_images = re.sub(r'\\label\{.*?\}', '', content_no_images)

    # 移除表格
    content_no_tables = re.sub(r'\\begin{table}.*?\\end{table}', '', content_no_images, flags=re.DOTALL)
    content_no_tables = re.sub(r'\\begin{table\*}.*?\\end{table\*}', '', content_no_tables, flags=re.DOTALL)

    # 移除参考文献
    content_no_tables = re.sub(r'\\begin{thebibliography}.*?\\end{thebibliography}', '', content_no_tables,
                               flags=re.DOTALL)

    # 移除 acknowledgments
    content_no_tables = re.sub(r'\\begin{acknowledgments}.*?\\end{acknowledgments}', '', content_no_tables,
                               flags=re.DOTALL)

    # 移除公式
    formulas_patterns = [
        r'\$.*?\$', r'\$\$.*?\$\$', r'\\$ .*?\\ $',
        r'\\begin{equation}.*?\\end{equation}',
        r'\\begin{align}.*?\\end{align}',
        # 其他公式环境
        r'\\begin{eqnarray}.*?\\end{eqnarray}',  # wumin
        r'\\begin{multline}.*?\\end{multline}',
    ]
    content_no_formulas = content_no_tables
    for pattern in formulas_patterns:
        content_no_formulas = re.sub(pattern, '', content_no_formulas, flags=re.DOTALL)

    # 移除空行
    content_no_formulas = re.sub(r'\n\s*\n', '\n', content_no_formulas)

    # 抽取标题
    title_match = re.search(r'\\title{(.*?)}', content_no_formulas)
    title = title_match.group(1) if title_match else 'No title found'

    # 抽取摘要
    abstract_match = re.search(r'\\begin{abstract}(.*?)\\end{abstract}', content_no_formulas, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else 'No abstract found'

    # 抽取章节
    sections = re.findall(r'\\section{(.*?)}(.*?)((?=\\section{)|\\z)', content_no_formulas, re.DOTALL)
    if 0 == len(sections):
        sections = extract_titles_and_contents(
            content_no_formulas)  # re.findall(r'\\textit{(.*?)}(.*?)((?=\\textit{)|\\z)', content_no_formulas, re.DOTALL)

    content_dict = {
        "title": title,
        "abstract": abstract,
        "sections": {}
    }

    for section in sections:
        section_title = section[0].strip()
        section_content = section[1].strip()

        # 定义替换规则：删除 \subsection{} 和 \subsubsection{}
        # 但保留花括号内的内容
        section_content = re.sub(r'\\subsection\{([^\}]+)\}', r'\1', section_content)
        section_content = re.sub(r'\\subsubsection\{([^\}]+)\}', r'\1', section_content)

        content_dict["sections"][section_title] = section_content

        # wumin
        # print(f"Title: {content_dict['title']}")
        # print(f"Abstract: {content_dict['abstract']}\n")
        #
        # for section_title, section_content in content_dict['sections'].items():
        #     print(f"Section: {section_title}")
        #     print(section_content)
        #     print("\n")

    strAllContent = ""
    strAllContent = strAllContent + content_dict['title'] + "\n" + content_dict['abstract'] + "\n"
    for section_title, section_content in content_dict['sections'].items():
        strAllContent = strAllContent + section_title + "\n" + section_content + "\n"

    return content_dict['title'], content_dict['abstract'],content_dict['sections'],strAllContent  # content_dict


def text_vector(text: str, file_name: str, file_path: str, paper_title):
    text_splitter = CharacterTextSplitter(
        separator=",",
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    docs = text_splitter.create_documents([text])
    print(f'文件{file_path}切分{len(docs)}个chunk')
    documents_vector_store = get_documents_vector_store()
    for doc in docs:
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "paper_title": paper_title
        }
        doc_with_metadata = Document(page_content=doc.page_content, metadata=metadata)

        documents_vector_store.add_documents([doc_with_metadata], ids=[uuid.uuid4()])


def get_documents_vector_store():
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        openai_api_key='c7f059b909884d7aa70b847d62a7b65f',
        openai_api_type='azure',
        openai_api_base='https://0518.openai.azure.com/'

    )
    return OpenSearchVectorSearch(opensearch_url="http://10.100.10.20:9200",
                                  index_name="semiconductor_test_vectors",
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


if __name__ == '__main__':
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_token = 0
    with open("arxiv7_complete_file.txt", "a+") as f:
        f.seek(0)
        x = f.readlines()
        complete_file=[k.replace("\n","") for k in x]
        pathList = find_files("/Users/zhangzhiyong/Documents/文档实验室/arxiv7", "tex")
        # for file_path in pathList:
        for index, file_path in enumerate(pathList):
            # # 先写100份
            # f.seek(0)
            # x = f.readlines()
            # complete_file = [k.replace("\n", "") for k in x]
            # if len(complete_file) == 10:
            #     break

            print(f'正在Embedding第{index+1}份文件')

            file_name = os.path.basename(file_path)
            if file_path not in complete_file:
                paper_title,paer_abstract,paer_sections, latex_content = parse_latex(file_path)

                if paer_abstract == 'No abstract found' and  paer_sections == {}:
                        print(f'文件{file_path}无法进行Embedding')
                        with open('arxiv7_complete_file_No_title.txt', 'a') as file:
                            file.write(file_path+' '+ str(index) + "\n")
                        f.write(file_path + "\n")
                        continue
                # TODO  paper_title != No title found   文章标题不用管
                #  paer_abstract != No abstract found or  paer_sections != {} 这两个只要有一个满足就可以做Embedding
                base_directory = '/Users/zhangzhiyong/Documents/文档实验室'
                relative_path = os.path.relpath(file_path, base_directory)
                data = latex_content
                token = encoding.encode(data)
                total_token += len(token)
                #text_vector(text=latex_content,file_name=file_name,file_path=relative_path,paper_title=paper_title)
                f.write(file_path+"\n")
                print(file_name,'\n')

            else:
                print(f"{file_name}已经存在，跳过")
    print(total_token)