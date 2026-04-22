#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RAG对话接口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # 回到src/rag目录
sys.path.append(str(project_root))

def test_chat_interface():
    """测试对话接口"""
    print("🧪 测试RAG对话接口")
    print("=" * 50)
    
    try:
        # 导入必要的模块
        from rag_system import MedicalRAG
        from chat.chat_interface import ChatInterface
        
        print("✅ 模块导入成功")
        
        # 初始化RAG系统（不构建索引）
        print("🔧 初始化RAG系统...")
        rag_system = MedicalRAG(chat=True)
        
        # 创建对话接口
        print("🔧 创建对话接口...")
        chat_interface = ChatInterface(rag_system)
        
        print("✅ 对话接口创建成功")
        
        # 测试单次对话
        print("\n💬 测试单次对话...")
        test_query = "什么是失眠？"
        response = chat_interface.chat_with_rag(test_query, k=3, use_hybrid_retrieval=True)
        
        print(f"问题：{test_query}")
        print(f"回答：{response}")
        
        # 启动交互式对话服务
        print("\n🚀 启动交互式对话服务...")
        print("提示：输入 'exit' 退出，输入 'help' 查看帮助")
        
        chat_interface.start_chat_service()
        
    except ImportError as e:
        print(f"❌ 导入失败：{e}")
        print("请确保所有依赖模块都已正确安装")
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试（不启动对话服务）"""
    print("🧪 快速测试RAG对话接口")
    print("=" * 50)
    
    try:
        from rag_system import MedicalRAG
        from chat.chat_interface import ChatInterface
        
        # 初始化系统
        rag_system = MedicalRAG(build_index=False)
        chat_interface = ChatInterface(rag_system)
        
        # 测试几个问题
        test_questions = [
            "失眠的症状有哪些？",
            "如何治疗失眠？",
            "失眠的预防方法是什么？"
        ]
        
        for question in test_questions:
            print(f"\n🔍 问题：{question}")
            response = chat_interface.chat_with_rag(question, k=3)
            print(f"🤖 回答：{response}...")
        
        print("\n✅ 快速测试完成")
        
    except Exception as e:
        print(f"❌ 快速测试失败：{e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        test_chat_interface()
