import os
import shutil
import time
import logging
import argparse
from typing import List

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("batch_mineru.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_unprocessed_files(input_dir: str, processed_files: set) -> List[str]:
    """获取未处理的文件列表"""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files
        if os.path.join(root, f) not in processed_files
    ]

def create_batch_folder(batch_files: List[str], batch_id: int, temp_dir: str) -> str:
    """创建临时批次文件夹并复制文件"""
    batch_folder = os.path.join(temp_dir, f"batch_{batch_id}")
    os.makedirs(batch_folder, exist_ok=True)
    
    for src_file in batch_files:
        shutil.copy(src_file, batch_folder)
        logger.debug(f"📄 已复制到批次文件夹: {os.path.basename(src_file)}")
    
    return batch_folder

def process_batch(batch_folder: str, output_dir: str) -> bool:
    """调用 mineru 处理整个批次文件夹"""
    try:
        cmd = f'mineru -p "{batch_folder}" -o "{output_dir}" -f false -t false'
        logger.info(f"🔧 执行命令: {cmd}")
        if os.system(cmd) == 0:  # 检查命令是否成功执行
            return True
    except Exception as e:
        logger.error(f"❌ 处理失败: {str(e)}")
    return False

def load_processed_files(log_file: str) -> set:
    """从日志加载已处理文件"""
    processed = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith("✅"):
                    processed.add(line.split(": ")[1].strip())
    return processed

def main(args):
    # 初始化目录
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "_batches_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # 加载已处理记录
    processed_files = load_processed_files(args.log_file)
    all_files = get_unprocessed_files(args.input_dir, processed_files)
    logger.info(f"📂 文件统计: 共 {len(all_files)} 个 | 已处理 {len(processed_files)} 个 | 待处理 {len(all_files)} 个")

    # 分批处理
    for batch_id, i in enumerate(range(0, len(all_files), args.batch_size), start=1):
        batch = all_files[i:i + args.batch_size]
        logger.info(f"\n🔄 正在处理批次 {batch_id} (共 {len(batch)} 个文件)")

        # 创建临时批次文件夹
        batch_folder = create_batch_folder(batch, batch_id, temp_dir)

        # 调用 mineru 处理
        if process_batch(batch_folder, args.output_dir):
            with open(args.log_file, 'a') as f:
                for file in batch:
                    f.write(f"✅ 处理成功: {file}\n")
            logger.info(f"🎉 批次 {batch_id} 处理完成!")
        else:
            logger.error(f"⚠️ 批次 {batch_id} 处理失败!")

        # 清理临时文件 (可通过 --keep-temp 保留)
        if not args.keep_temp:
            shutil.rmtree(batch_folder, ignore_errors=True)

        # 批次间隔
        if i + args.batch_size < len(all_files) and args.delay > 0:
            logger.info(f"⏸ 等待 {args.delay} 秒...")
            time.sleep(args.delay)

    # 清理临时根目录
    if not args.keep_temp and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mineru 批次文件夹处理器")
    parser.add_argument("-i", "--input-dir", required=True, help="输入文件夹路径")
    parser.add_argument("-o", "--output-dir", required=True, help="输出文件夹路径")
    parser.add_argument("-b", "--batch-size", type=int, default=10, help="每批处理文件数 (建议5-20)")
    parser.add_argument("--log-file", default="processed.log", help="日志文件路径")
    parser.add_argument("--delay", type=int, default=3, help="批次间隔秒数")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时批次文件夹 (调试用)")
    args = parser.parse_args()

    logger.info(f"⚡ 启动 Mineru 批次处理器 (每批 {args.batch_size} 个文件)")
    main(args)
