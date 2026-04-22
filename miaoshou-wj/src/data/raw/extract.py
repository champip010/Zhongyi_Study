import os
import shutil

# 父目录路径（请修改为实际路径）
parent_dir = r"./相关书籍"  # 例如 "/home/user/study/rag/1/专家共识与指南"

# 遍历所有文件
for root, dirs, files in os.walk(parent_dir, topdown=False):  # bottom-up 防止先删除顶层
    for file in files:
        if file.endswith(".md"):  # 只处理 markdown 文件
            file_path = os.path.join(root, file)
            
            # 获取最顶层子目录名（即 parent_dir 下的第一层文件夹名）
            rel_path = os.path.relpath(file_path, parent_dir)
            top_folder = rel_path.split(os.sep)[0]  # 第一层文件夹名
            
            # 构造新文件名
            new_name = top_folder + ".md"
            new_path = os.path.join(parent_dir, new_name)
            
            # 如果已存在，添加序号避免覆盖
            counter = 1
            while os.path.exists(new_path):
                new_name = f"{top_folder}_{counter}.md"
                new_path = os.path.join(parent_dir, new_name)
                counter += 1
            
            # 移动文件
            shutil.move(file_path, new_path)
            print(f"移动: {file_path} → {new_path}")
    
    # 删除空文件夹（除了 parent_dir 自身）
    if root != parent_dir and not os.listdir(root):
        os.rmdir(root)
        print(f"删除空文件夹: {root}")
