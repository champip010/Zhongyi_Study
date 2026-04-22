import os
import re
import random
import string
import shutil

def clean_filename(filename):
    """
    清理文件名，只保留字母、中文和数字
    :param filename: 原始文件名
    :return: 清理后的文件名（不含扩展名），扩展名
    """
    # 分离文件名和扩展名
    basename, ext = os.path.splitext(filename)
    
    # 匹配字母、中文、数字
    # \w 匹配字母、数字和下划线，\u4e00-\u9fff 是中文范围
    pattern = re.compile(r'[^\w\u4e00-\u9fff]', re.UNICODE)
    cleaned = re.sub(pattern, '', basename).strip()
    
    return cleaned, ext.lower()

def generate_alnum_name(length=8):
    """生成只包含字母和数字的随机文件名"""
    chars = string.ascii_letters + string.digits  # 大小写字母+数字
    return ''.join(random.choices(chars, k=length))

def process_directory(parent_dir):
    """
    处理目录中的所有文件
    :param parent_dir: 父目录路径
    """
    parent_dir = os.path.abspath(parent_dir)
    used_names = set()
    
    # 先收集所有文件路径
    all_files = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # 处理每个文件
    for filepath in all_files:
        # 获取相对路径（用于判断是否需要移动）
        relpath = os.path.relpath(filepath, parent_dir)
        
        # 清理文件名
        original_name = os.path.basename(filepath)
        cleaned, ext = clean_filename(original_name)
        
        # 如果清理后名称为空或不包含合法字符，生成随机名称
        if not cleaned:
            cleaned = generate_alnum_name()
        
        # 确保名称唯一（不区分大小写）
        new_name = f"{cleaned}{ext}"
        base_name = cleaned
        counter = 1
        while new_name.lower() in used_names:
            new_name = f"{base_name}_{counter}{ext}"
            counter += 1
        
        used_names.add(new_name.lower())
        
        # 目标路径（都在父目录下）
        dest_path = os.path.join(parent_dir, new_name)
        
        # 如果不是在父目录下或者需要重命名，则移动文件
        if relpath != new_name or os.path.dirname(filepath) != parent_dir:
            print(f"处理: {filepath} -> {dest_path}")
            
            # 处理文件冲突
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            # 移动文件
            shutil.move(filepath, dest_path)
    
    # 删除空子目录
    for root, dirs, _ in os.walk(parent_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
                print(f"删除空目录: {dir_path}")
            except OSError:
                pass  # 目录不为空则不删除

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="清理文件名并将所有文件移动到父目录")
    parser.add_argument("directory", help="要处理的父目录路径")
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误: {args.directory} 不是有效目录")
        exit(1)
    
    print(f"开始处理目录: {args.directory}")
    process_directory(args.directory)
    print("处理完成")
