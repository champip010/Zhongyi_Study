#!/usr/bin/env python3
"""
测试分层索引功能
"""
import os

# 设置环境变量
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

from rag_system import MedicalRAG

def test_hierarchical_index():
    """测试分层索引构建和检索"""
    print("🚀 开始测试分层索引...")
    
    # 创建 RAG 系统
    rag = MedicalRAG(load_llm=False)
    
    # 构建分层索引
    print("\n📚 构建分层索引...")
    rag.build_index()
    
    # 测试不同类型的查询
    test_queries = [
        "失眠的诊断标准是什么？",           # 诊断类查询
        "如何治疗高血压？",                # 治疗类查询
        "糖尿病患者的饮食注意事项",         # 一般查询
        "心绞痛的典型症状表现",            # 症状类查询
        "阿司匹林的副作用有哪些？"         # 药物类查询
    ]
    
    print("\n🔍 测试分层检索...")
    for query in test_queries:
        print(f"\n查询: {query}")
        results = rag.search(query, k=3)
        
        print(f"找到 {len(results)} 个结果:")
        for i, doc in enumerate(results):
            level = doc.metadata.get('level', 'unknown')
            score = doc.metadata.get('score', 0)
            source = doc.metadata.get('source', 'unknown')
            print(f"  {i+1}. [{level}] {source} (分数: {score:.3f})")
            print(f"     内容: {doc.page_content[:100]}...")

if __name__ == "__main__":
    test_hierarchical_index()
