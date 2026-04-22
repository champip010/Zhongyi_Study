#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量更新系统设置文件
用户可以在这里修改关键参数
"""

# 数据库设置
COLLECTION_NAME = "medical_knowledge"  # Qdrant集合名称
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # Embedding模型
DEVICE = "cuda:0"  # 计算设备 (cuda:0, cuda:1, cpu)

# 文档处理设置
CHUNK_SIZE = 800  # 文档分块大小
CHUNK_OVERLAP = 200  # 分块重叠大小

# 功能开关
UPDATE_HYBRID = True  # 是否更新混合搜索集合
REBUILD_CONNECTIONS = True  # 是否构建知识连接

# 去重设置
SIMILARITY_THRESHOLD = 0.85  # 去重相似度阈值



