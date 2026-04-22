# RAG系统使用说明

## 目录
1. [如何使用search功能](#1-如何使用search功能)-最主要功能，实现文档检索
2. [Qdrant数据库管理](#2-qdrant数据库管理)
3. [如何增加新的文件](#3-如何增加新的文件)
4. [如何使用chat文件夹测试检索效果](#4-如何使用chat文件夹测试检索效果)
5. [使用到的模型和需要的资源](#5-使用到的模型和需要的资源)

将rag文件夹放在和llm文件夹同一位置下，初始化rag然后修改qa_base中的search_docs即可：

```python
from rag.rag_system import MedicalRAG

self.rag = MedicalRAG()

async def generate_stream():
    ...
    ...
    search_docs = self.rag.search(query=new_question)

```
---

## 1. 如何使用search功能

### 1.1 基本使用方法

```python
from rag_system import MedicalRAG

# 初始化RAG系统
rag = MedicalRAG()

# 基础搜索
docs = rag.search(query="什么是糖尿病？", k=5) 

# 混合检索搜索
docs = rag.search(query="什么是糖尿病？", k=5, use_hybrid_retrieval=True)
```

### 1.2 docs的类型和返回值
返回文件为langchain的Document类型，包括：
- page_content（str）：块内容
- metadata：元数据
    - 'has_diagnosis'，'has_treatment'，'has_evaluation'，'has_standards'，'has_titles'（bool）: 是否有某些实体
    - 'chunk_position'（int）: 在文档中的位置
    - 'content_type'（str）: 类型分类
    - 'source'（str）: 来源的文档名
    - 'chunk_type'（str）: 块类型 
    - 'relative_path'（str）: 相对文件夹位置
    - 'connections'（json类型，使用前需要先转换类型，转换后为list）: 知识连接到的chunk
        - 'content_type'（str）: 块类型
        - 'similarity'（float）: 相似度 
        - 'doc_id'（int）: 块id 
    - 'medical_entities'（str）: 是否有某个病的名称，如：糖尿病
    - 'folder'（str）: 文件夹 
    - 'file_name'（str）: 文件名

### 1.3 search方法主要参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | 必需 | 用户查询问题 |
| `k` | int | 5 | 返回的文档数量 |
| `use_hybrid_retrieval` | bool | False | 是否使用混合检索 |

### 1.4 检索策略说明

系统采用多级检索策略，按优先级顺序：

1. **混合检索** (`use_hybrid_retrieval=True`)
   - 结合语义检索和BM25检索
   - 使用colbertv2.0进行嵌入和排序

2. **多跳检索** (默认)
   - 基于HyDE增强查询
   - 利用知识连接进行扩展检索
   - 使用交叉编码器重排序

3. **基础检索**
   - 直接使用Qdrant向量数据库检索
   - 作为兜底方案
   - 如何使用不了多跳系统会自动跳转到基础检索

### 1.4 如何更换模型
当前使用的模型都已经下载到：/home/wangjun/.cache/huggingface/hub

#### 更换LLM模型

```python
# 在初始化时指定不同的LLM模型
rag = MedicalRAG(llm_model = "Qwen/Qwen2.5-7B-Instruct",embedding_model_name = "BAAI/bge-large-zh-v1.5",
            rerank_model_name = "BAAI/bge-reranker-large")
            
- 使用vllm加载llm模型
- SentenceTransformer和加载embedding
- CrossEncoder加载重排模型

```

---

## 2. Qdrant数据库管理

### 数据库部署与连接
使用docker本地部署qdrant数据库
```bash

docker run --name=qdrant --volume /home/wangjun/study/rag/wj-miaoshou/ratubrain-semiconductor/app/data/app/data/qdrant_data:/qdrant/storage --network=bridge --workdir=/qdrant -p 6333:6333 -p 6334:6334 --runtime=runc 

```

数据已做本地持久化，位置在：/home/wangjun/study/rag/wj-miaoshou/ratubrain-semiconductor/app/data/app/data/qdrant_data


```python
from QdrantManager import QdrantManager

# 基础配置
qdrant = QdrantManager(
    collection_name="medical_knowledge",  # 集合名称
    host="127.0.0.1",                    # 数据库地址
    port=6333,                           # 端口号
    vector_size=1024,                    # 向量维度
    embedding_model_name="BAAI/bge-large-zh-v1.5",  # 嵌入模型
    embedding_device=None            # 自动选择空闲设备
)
```
### 当前数据库

数据库当前一共有两个表（collection）分别为：
- medical_knowledge存储多跳数据库：单一密集向量
- hybrid-search混合检索的数据库：多向量类型


---

## 3. 如何增加新的文件

### 3.1 文件格式要求

系统支持以下格式：
- **Markdown (.md)** 
- **文本文件 (.txt/.docx)** 
- **pdf** 

### 3.2 增量更新

使用incremental_update下的工具

```bash
# 指定新数据文件夹路径
python run_update.py <新数据文件夹路径>

# 示例
python run_update.py ../data/new_medical_data/
python run_update.py /path/to/your/new/data/
```

- run_update.py会自动更新已经运行的qdrant数据库
- 用来增量更新数据库，会自动处理新旧chunk间的知识连接

---

## 4. 如何和模型对话测试效果

使用chat文件夹下的test_chat.py文件

```bash
# 指定新数据文件夹路径
python test_chat.py 
```
- 这里的回答使用的是MedicalRAG初始化的llm
- 可以用help参考可以修改的操作
- 支持10条的历史信息

---

## 5. 使用到的模型和需要的资源

### 5.1 核心模型列表

#### LLM模型
- **主模型**：`Qwen/Qwen2-VL-7B-Instruct`


#### Embedding模型
- **主模型**：`BAAI/bge-large-zh-v1.5` (1024维)


#### 重排序模型
- **主模型**：`BAAI/bge-reranker-large`


### 5.2 硬件资源要求

主要需要的资源为显存，而对显存占比最大的是加载LLM主模型，如果后续替换成使用本地部署的别的模型显存需求会很小
如果不替换至少需要2张gpu来加载llm，reranker和embedding模型


---

