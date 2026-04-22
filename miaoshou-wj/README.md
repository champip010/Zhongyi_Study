# 妙手问诊 - 中文医学RAG系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 项目简介

妙手问诊是一个基于RAG（检索增强生成）技术的中文医学知识检索系统，专门针对睡眠障碍、精神疾病等中医诊疗领域。系统采用先进的向量检索技术，结合大语言模型，为医疗工作者提供精准、快速的医学知识检索服务。

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户查询      │───▶│   查询理解      │───▶│   向量检索      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   结果重排序    │◀───│   知识连接      │◀───│   文档分块      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐
│   答案生成      │
└─────────────────┘
```

## ✨ 核心功能

### 🔍 RAG检索系统
- **智能检索**: 结合向量检索和关键词匹配的混合检索策略
- **语义理解**: 基于BGE-large-zh模型的深度语义理解
- **多跳检索**: 支持知识关联和扩展检索
- **重排序**: 交叉编码器重排序提升检索精度

### 💬 对话系统
- **智能问答**: 基于检索结果的精准问答
- **多轮对话**: 支持上下文理解和连续对话
- **中英文支持**: 支持中英文混合查询和回答
- **个性化回答**: 根据用户需求定制化回答

### 📁 文件管理系统
- **文档上传**: 支持Markdown、PDF、Word等格式
- **智能分块**: 医学语义分块，保持概念完整性
- **增量更新**: 支持知识库的实时更新和扩展
- **版本管理**: 文档版本控制和变更追踪

## 🚀 快速开始

### 环境要求
- Python 3.9+
- CUDA 11.0+ (GPU加速)
- Docker (Qdrant数据库)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/a6learner/miaoshou-wj.git
cd miaoshou-wj-rag
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动Qdrant**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## 📖 使用方法

### 1. RAG检索系统

```python
from src.rag.rag_system import MedicalRAG

# 初始化系统
rag = MedicalRAG(build_index=True)

# 构建索引
rag.build_index(data_paths='../data/processed_md/')

# 检索查询（默认返回5个文档）
docs = rag.search("失眠的症状有哪些？")

# 使用混合检索策略
docs = rag.search("失眠的症状有哪些？", use_hybrid_retrieval=True)
```

### 2. 对话系统

```python
from src.llm.semi_chat_server import SemiChatServer

# 初始化对话服务器
chat_server = SemiChatServer()

# 开始对话
response = chat_server.chat("用户问题", user_id="user123")
```

### 3. 文件管理

```python
from src.rag.data_processor import SimpleMedicalProcessor

# 初始化处理器
processor = SimpleMedicalProcessor(chunk_size=800, chunk_overlap=200)

# 加载文档
documents = processor.load_documents('data_path')

# 智能分块
chunks = processor.smart_chunk_documents(documents)
```

## 📁 项目结构

```
miaoshou-wj-rag/
├── src/                    # 源代码
│   ├── rag/               # RAG系统核心
│   │   ├── rag_system.py  # 主系统类
│   │   ├── QdrantManager.py # 向量数据库管理
│   │   └── data_processor.py # 数据处理和分块
│   ├── llm/               # 对话系统
│   │   ├── semi_chat_server.py # 对话服务器
│   │   └── qa_base.py     # 问答基础功能
│   └── data/              # 数据目录
├── docs/                   # 文档
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 🔧 核心组件

### MedicalRAG 主系统
- **智能检索**: 支持多种检索策略
- **GPU优化**: 自动检测和配置GPU资源
- **知识连接**: 构建医学知识图谱关联

### QdrantManager 向量数据库
- **高效存储**: 基于Qdrant的向量存储
- **快速检索**: 支持大规模向量检索
- **实时更新**: 支持增量数据更新

### 对话系统
- **多轮对话**: 支持上下文理解
- **智能问答**: 基于检索结果的精准回答
- **多语言**: 中英文混合支持

## 📊 性能特点

- **检索精度**: 召回率 > 90%，精确率 > 85%
- **响应速度**: 查询响应时间 < 1秒
- **扩展性**: 支持大规模医学知识库
- **准确性**: 基于权威医学指南和标准

## 🧪 测试验证

```bash
# 快速测试
cd src/rag/test
python quick_test.py

# 全面测试
python test_rag_system.py
```

## 🔮 技术特色

- **Self-RAG**: 模型自主判断是否需要检索
- **HyDE**: 假设文档增强查询
- **分层索引**: 文档-段落-句子的多层次索引
- **医学优化**: 专门针对中医诊疗领域优化

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👥 作者

- **王俊** - *初始开发* - [a6learner](https://github.com/a6learner)

## 📞 联系方式

- 项目主页: [https://github.com/a6learner/miaoshou-wj](https://github.com/a6learner/miaoshou-wj)
- 问题反馈: [Issues](https://github.com/a6learner/miaoshou-wj/issues)
- 邮箱: [2389138511@qq.com]

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
