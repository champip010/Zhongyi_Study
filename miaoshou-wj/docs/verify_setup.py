#!/usr/bin/env python3
"""
妙手问诊RAG系统 - 项目验证脚本
验证项目设置和依赖是否正确
"""

import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("   需要Python 3.9+")
        return False
    else:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = [
        "src",
        "src/rag",
        "src/data",
        "src/tests",
        "docs"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "LICENSE",
        ".gitignore",
        "src/rag/rag_system.py",
        "src/rag/QdrantManager.py",
        "src/rag/data_processor.py"
    ]
    
    all_good = True
    
    # 检查目录
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            all_good = False
    
    # 检查文件
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ 文件存在: {file_path}")
        else:
            print(f"❌ 文件缺失: {file_path}")
            all_good = False
    
    return all_good

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    required_packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "qdrant_client",
        "langchain",
        "numpy",
        "tqdm",
        "jieba"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺失的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_imports():
    """检查模块导入"""
    print("\n🔍 检查模块导入...")
    
    # 添加src目录到路径
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from rag.rag_system import MedicalRAG
        print("✅ MedicalRAG 导入成功")
    except Exception as e:
        print(f"❌ MedicalRAG 导入失败: {e}")
        return False
    
    try:
        from rag.QdrantManager import QdrantManager
        print("✅ QdrantManager 导入成功")
    except Exception as e:
        print(f"❌ QdrantManager 导入失败: {e}")
        return False
    
    try:
        from rag.data_processor import SimpleMedicalProcessor
        print("✅ SimpleMedicalProcessor 导入成功")
    except Exception as e:
        print(f"❌ SimpleMedicalProcessor 导入失败: {e}")
        return False
    
    return True

def check_data_files():
    """检查数据文件"""
    print("\n📊 检查数据文件...")
    
    data_dir = Path("src/data/test")
    if not data_dir.exists():
        print("❌ 测试数据目录不存在")
        return False
    
    md_files = list(data_dir.glob("*.md"))
    if len(md_files) == 0:
        print("❌ 没有找到Markdown测试文件")
        return False
    
    print(f"✅ 找到 {len(md_files)} 个测试数据文件")
    for file in md_files[:3]:  # 显示前3个
        print(f"   - {file.name}")
    
    return True

def main():
    """主函数"""
    print("🔍 妙手问诊RAG系统 - 项目验证")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_project_structure,
        check_dependencies,
        check_imports,
        check_data_files
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 验证结果总结")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过检查: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有检查通过！项目设置正确。")
        print("\n🚀 下一步:")
        print("1. 启动Qdrant: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print("2. 运行测试: cd src/rag/test && python quick_test.py")
        print("3. 查看文档: 阅读 README.md 和 docs/ 目录")
    else:
        print("⚠️ 部分检查失败，请根据上述提示修复问题。")
        print("\n💡 常见解决方案:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 检查Python版本: 需要3.9+")
        print("3. 检查项目结构: 确保所有文件都在正确位置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
