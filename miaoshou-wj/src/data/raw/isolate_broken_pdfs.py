#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# 确保正确定义PdfDocument
try:
    from pypdfium2 import PdfDocument
    from pypdfium2._helpers.misc import PdfiumError
except ImportError as e:
    raise RuntimeError("请先安装pypdfium2库: pip install --upgrade pypdfium2") from e

def validate_pdf(filepath):
    """多层级PDF验证"""
    try:
        # 基础校验
        with open(filepath, 'rb') as f:
            header = f.read(1024)
            if not header.startswith(b'%PDF-'):
                return False
        
        # 深度校验
        doc = PdfDocument(filepath)
        if len(doc) == 0:
            return False
        # 验证第一页可访问
        page = doc[0]
        _ = page.get_size()
        return True
        
    except (PdfiumError, RuntimeError, ValueError) as e:
        print(f"确认损坏: {filepath}")
        return False
    except Exception as e:
        print(f"验证跳过: {filepath} ({type(e).__name__})")
        return True  # 非PDF问题，视为有效

def main(source_dir, broken_dir="broken_files"):
    source_path = Path(source_dir).absolute()
    broken_path = source_path / broken_dir
    
    broken_files = []
    for pdf_file in source_path.glob("**/*.pdf"):
        if not validate_pdf(pdf_file):
            target = broken_path / pdf_file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pdf_file), str(target))
            broken_files.append(pdf_file.name)
    
    print(f"\n处理完成！真实损坏文件: {len(broken_files)}个")
    if broken_files:
        print("隔离文件列表:")
        print("\n".join(f" - {f}" for f in broken_files))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="源目录路径")
    parser.add_argument("--broken-dir", default="broken_files", help="损坏文件目录名")
    args = parser.parse_args()
    
    main(args.source_dir, args.broken_dir)
