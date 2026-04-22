#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统增量更新工具
简单易用，只需要指定新数据文件夹路径
"""


import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rag.incremental_update.incremental_updater import IncrementalUpdater
from rag.QdrantManager import QdrantManager

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🚀 RAG系统增量更新工具")
        print("=" * 50)
        print("使用方法：")
        print("  python run_update.py <新数据文件夹路径>")
        print("")
        print("示例：")
        print("  python run_update.py ../data/new_medical_data/")
        print("  python run_update.py /path/to/your/new/data/")
        return
    
    data_path = sys.argv[1]
    
    print("🚀 RAG系统增量更新工具")
    print("=" * 50)
    print(f"📁 新数据路径：{data_path}")
    
    try:
        # 1. 检查新数据路径
        if not os.path.exists(data_path):
            print(f"❌ 新数据路径不存在：{data_path}")
            return
        
        # 2. 加载设置
        try:
            import settings
            print("✅ 已加载设置文件")
            COLLECTION_NAME = getattr(settings, 'COLLECTION_NAME', "medical_knowledge")
            EMBEDDING_MODEL = getattr(settings, 'EMBEDDING_MODEL', "BAAI/bge-large-zh-v1.5")
            CHUNK_SIZE = getattr(settings, 'CHUNK_SIZE', 600)
            CHUNK_OVERLAP = getattr(settings, 'CHUNK_OVERLAP', 150)
            UPDATE_HYBRID = getattr(settings, 'UPDATE_HYBRID', True)
            REBUILD_CONNECTIONS = getattr(settings, 'REBUILD_CONNECTIONS', True)
        except ImportError:
            print("⚠️ 未找到设置文件，使用默认配置")

        
        # 3. 初始化Qdrant管理器
        print("🔧 初始化Qdrant管理器...")
        qdrant_manager = QdrantManager(
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL,
        )
        
        # 4. 创建增量更新器
        print("🔧 创建增量更新器...")
        updater = IncrementalUpdater(
            qdrant_manager=qdrant_manager,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # 5. 检查系统状态
        print("📊 检查系统状态...")
        status = updater.get_update_status()
        print(f"主集合文档数：{status['main_collection']['document_count']}")
        print(f"混合搜索集合文档数：{status['hybrid_collection']['document_count']}")
        
        # 6. 执行增量更新
        print("🚀 开始执行增量更新...")
        result = updater.add_new_data(
            new_data_paths=data_path,
            update_hybrid=UPDATE_HYBRID,
            rebuild_connections=REBUILD_CONNECTIONS
        )
        
        # 7. 显示更新结果
        print("\n" + "=" * 50)
        print("🎉 更新完成！")
        print(f"新增文档数：{result['added_count']}")
        print(f"总文档数：{result['total_count']}")
        
    except Exception as e:
        print(f"❌ 更新过程中发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
