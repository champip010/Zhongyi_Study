import os
import zipfile
import rarfile

def unzip_all_in_folder(folder_path, exclude_folder='chenmingle'):
    for root, dirs, files in os.walk(folder_path):
        if exclude_folder in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.rar'):
                rar_path = os.path.join(root, file)
                extract_path = root
                # 使用7z命令行工具解压rar文件
                cmd = f'7z x "{rar_path}" -o"{extract_path}" -y'
                result = os.system(cmd)
                if result == 0:
                    print(f"解压完成: {rar_path}")
                else:
                    print(f"解压失败: {rar_path}")

if __name__ == "__main__":
    base_folder = input("请输入要解压缩的文件夹路径: ")
    unzip_all_in_folder(base_folder)

