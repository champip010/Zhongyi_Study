import os
from collections import defaultdict

def count_files(directory):
    """
    统计文件夹下的文件数量（同名文件只计一次）和扩展名分布
    :param directory: 目标文件夹路径
    :return: (总文件数, {扩展名: 数量})
    """
    seen_files = set()  # 记录已统计过的文件名
    ext_stats = defaultdict(int)

    for root, _, files in os.walk(directory):
        for file in files:
            if file not in seen_files:  # 避免同名文件重复统计
                seen_files.add(file)
                _, ext = os.path.splitext(file)
                ext = ext.lower()  # 统一转为小写
                ext_stats[ext] += 1

    total = len(seen_files)
    return total, dict(ext_stats)

if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="统计文件夹下的文件数量（同名只算一次）")
    parser.add_argument("directory", help="要统计的文件夹路径")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误: {args.directory} 不是有效目录")
        exit(1)

    total, ext_stats = count_files(args.directory)

    # 打印结果
    print(f"\n统计结果: {args.directory}")
    print(f"总唯一文件名数: {total}")
    print("\n按扩展名统计（基于唯一文件名）:")
    for ext, count in sorted(ext_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext if ext else '[无扩展名]'}: {count} 个")
