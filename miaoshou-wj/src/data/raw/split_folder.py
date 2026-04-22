import os
import shutil

def split_folder(base_folder, max_files=100):
    # 获取所有文件（递归）
    all_files = []
    for root, dirs, files in os.walk(base_folder):
        for f in files:
            abs_path = os.path.join(root, f)
            all_files.append(abs_path)

    # 目标根目录
    parent_dir = os.path.dirname(base_folder.rstrip('/'))
    base_name = os.path.basename(base_folder.rstrip('/'))

    # 分组并移动
    for idx, start in enumerate(range(0, len(all_files), max_files), 1):
        group_files = all_files[start:start+max_files]
        group_dir = os.path.join(parent_dir, f"{base_name}_part{idx}")
        os.makedirs(group_dir, exist_ok=True)
        for file_path in group_files:
            fname = os.path.basename(file_path)
            target_path = os.path.join(group_dir, fname)
            # 避免覆盖
            if os.path.abspath(file_path) != os.path.abspath(target_path):
                shutil.move(file_path, target_path)
        print(f"分组{idx}: {len(group_files)}个文件 -> {group_dir}")

    print(f"总共分为{(len(all_files)-1)//max_files+1}组")

if __name__ == "__main__":
    # 修改为你的实际路径
    base_folder = "./专家共识与指南"
    split_folder(base_folder, max_files=100)
