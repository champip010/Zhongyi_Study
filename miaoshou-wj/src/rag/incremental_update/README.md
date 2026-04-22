# RAG系统增量更新使用说明

## 概述

这是一个简单的RAG系统增量更新工具，用于向现有的RAG数据库添加新数据。

## 功能特点

- ✅ 自动处理新数据（加载、清洗、分块）
- ✅ 智能去重（基于语义相似度）
- ✅ 构建知识连接
- ✅ 更新混合搜索集合
- ✅ 自动更新已有chunk的连接状态
- ✅ 简单配置（通过settings.py文件）
- ✅ 一键运行（只需指定数据路径）
- ✅ 自动GPU检测（智能选择空闲显卡）

## 使用方法

### 1. 基本使用

```bash
# 指定新数据文件夹路径
python run_update.py <新数据文件夹路径>

# 示例
python run_update.py ../data/new_medical_data/
python run_update.py /path/to/your/new/data/
```

### 2. 自定义设置

如果需要修改参数，可以编辑 `settings.py` 文件：

```python
# 修改数据库设置
COLLECTION_NAME = "my_collection"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
# DEVICE = "cuda:1"  # 注释掉这行，启用自动检测

# 修改文档处理参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# 关闭某些功能
UPDATE_HYBRID = False # 是否更新混合知识库
REBUILD_CONNECTIONS = False # 是否建立知识连接
```

### 2. 数据要求

- 支持格式：`.md`, `.txt`, `.docx`
- 文件编码：UTF-8
- 文件大小：建议小于50MB

### 3. 系统要求

- Qdrant服务正在运行
- 已存在的RAG数据库
- GPU支持（推荐）

## 工作流程

1. **数据加载** → 扫描指定文件夹
2. **预处理** → 清洗和标准化
3. **分块** → 智能文档分块
4. **去重** → 避免重复数据
5. **向量化** → 生成embeddings
6. **入库** → 添加到主集合
7. **知识连接** → 构建文档关联
8. **混合搜索** → 更新搜索集合

## 注意事项

- 确保Qdrant服务正在运行
- 新数据路径必须存在
- 系统会自动处理所有支持格式的文件
- 更新过程可能需要一些时间，请耐心等待

## 故障排除

### 常见问题

1. **路径不存在**
   - 检查新数据文件夹路径是否正确
   - 确保文件夹包含文件

2. **Qdrant连接失败**
   - 检查Qdrant服务是否运行
   - 确认端口6333是否可访问

3. **GPU内存不足**
   - 减少批处理大小
   - 使用CPU模式（修改device参数）

## 示例

```bash
# 添加新的医学文档
python run_update.py ../data/new_medical_papers/

# 添加新的研究资料
python run_update.py /home/user/research_data/
```

## 技术支持

如果遇到问题，请检查：
1. 错误日志信息
2. 系统状态
3. 文件路径和权限
4. Qdrant服务状态

---

