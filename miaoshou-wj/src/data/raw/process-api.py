
import os
import re
import requests
import time
import zipfile


header = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI0OTYwODEwOSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1NDUzNzU5MiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiZmQ5YjZlMTEtM2I5Zi00ODZjLTg4MjUtZDRkYjdjOWQzNjE5IiwiZW1haWwiOiIiLCJleHAiOjE3NTU3NDcxOTJ9.GmNcC1Rm37F7vAyd6No5Ckfbi_yR2z4kKVJFfT-9F-_pT0cJHxOTUeL8OqD__amsWG_sSDLcrPVKjSTXfQYDmg"
    }

# def upload(has_upload: bool, base_folder: str):
#     if not has_upload:
#         url_post = "https://mineru.net/api/v4/file-urls/batch"
#         # 目标文件夹
#         base_file_path = base_folder.strip()
#         # 递归获取所有 PDF 和 DOCX 文件
#         all_files = []
#         for root, dirs, files in os.walk(base_file_path):
#             for f in files:
#                 if f.lower().endswith(".pdf") or f.lower().endswith(".docx") or f.lower().endswith(".doc"):
#                     abs_path = os.path.join(root, f)
#                     all_files.append(abs_path)

#         import hashlib
#         def make_valid_id(abs_path, base_file_path):
#             rel_path = os.path.relpath(abs_path, base_file_path)
#             hash_part = hashlib.sha1(rel_path.encode('utf-8')).hexdigest()[:16]
#             f = os.path.basename(abs_path)
#             legal_name = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(f)[0])
#             valid_id = f"{legal_name}_{hash_part}"
#             return valid_id[:128]

#         # 分批处理，每批最多30个文件
#         batch_ids = []
#         log_dir = os.path.join('log', os.path.basename(base_folder))
#         os.makedirs(log_dir, exist_ok=True)
#         log_path = os.path.join(log_dir, 'upload_fail.log')
#         import logging
#         logging.basicConfig(filename=log_path, level=logging.INFO)
#         for batch_idx, start in enumerate(range(0, len(all_files), 30), 1):
#             batch_files = all_files[start:start+30]
#             files_param = []
#             for abs_path in batch_files:
#                 f = os.path.basename(abs_path)
#                 valid_id = make_valid_id(abs_path, base_file_path)
#                 files_param.append({"name": f, "is_ocr": True, "data_id": valid_id})
#             data = {
#                 "enable_formula": False,
#                 "language": "auto",
#                 "enable_table": False,
#                 "files": files_param,
#             }
#             try:
#                 response = requests.post(url_post, headers=header, json=data)
#                 if response.status_code == 200:
#                     result = response.json()
#                     print(f"response success")
#                     if result["code"] == 0:
#                         batch_id = result["data"]["batch_id"]
#                         batch_ids.append(batch_id)
#                         urls = result["data"]["file_urls"]
#                         print(f"batch_id:{batch_id}")
#                         batch_id_path = os.path.join(log_dir, f'batch_id_{batch_idx}.txt')
#                         with open(batch_id_path, 'w', encoding='utf-8') as f:
#                             f.write(str(batch_id))
#                         failed_num = 0
#                         for i in range(len(urls)):
#                             file_name = os.path.basename(batch_files[i])
#                             success = False
#                             with open(batch_files[i], "rb") as f:
#                                 res_upload = requests.put(urls[i], data=f)
#                             if res_upload.status_code == 200:
#                                 print(f"{file_name} upload success")
#                                 success = True
#                             else:
#                                 print(f"{file_name} upload failed")
#                             if not success:
#                                 failed_num += 1
#                                 logging.info(f"{file_name} upload failed ")
#                         logging.info(f"Batch {batch_idx}: Total failed uploads: {failed_num},Total files: {len(urls)}")
#                         # 保存batch_id到log目录
                        
#                     else:
#                         print(f"apply upload url failed,reason:{result['msg']}")
#                 else:
#                     print(f"response not success. status:{response.status_code} ,result:{response}")
#             except Exception as err:
#                 print(err)
#         return batch_ids
#     else:
#         return ""

def upload(has_upload: bool, base_folder: str, max_retries=3, retry_interval=10):
    if not has_upload:
        url_post = "https://mineru.net/api/v4/file-urls/batch"
        base_file_path = base_folder.strip()
        
        # 递归获取所有 PDF 和 DOCX 文件
        all_files = []
        for root, dirs, files in os.walk(base_file_path):
            for f in files:
                if f.lower().endswith((".pdf", ".docx", ".doc")):
                    abs_path = os.path.join(root, f)
                    all_files.append(abs_path)

        import hashlib
        def make_valid_id(abs_path, base_file_path):
            rel_path = os.path.relpath(abs_path, base_file_path)
            hash_part = hashlib.sha1(rel_path.encode('utf-8')).hexdigest()[:16]
            f = os.path.basename(abs_path)
            legal_name = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(f)[0])
            valid_id = f"{legal_name}_{hash_part}"
            return valid_id[:128]

        batch_ids = []
        log_dir = os.path.join('log', os.path.basename(base_folder))
        os.makedirs(log_dir, exist_ok=True)
        upload_fail_log = os.path.join(log_dir, 'upload_fail.log')
        retry_log = os.path.join(log_dir, 'upload_retry.log')
        import logging
        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(upload_fail_log),
                logging.StreamHandler()
            ]
        )

        # 创建重试队列
        retry_queue = []
        
        for batch_idx, start in enumerate(range(0, len(all_files), 30), 1):
            batch_files = all_files[start:start+30]
            files_param = []
            for abs_path in batch_files:
                f = os.path.basename(abs_path)
                valid_id = make_valid_id(abs_path, base_file_path)
                files_param.append({"name": f, "is_ocr": True, "data_id": valid_id})
            
            data = {
                "enable_formula": False,
                "language": "auto",
                "enable_table": False,
                "files": files_param,
            }
            
            # 获取上传URL的批次ID
            batch_id = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(url_post, headers=header, json=data)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("code") == 0:
                            batch_id = result["data"]["batch_id"]
                            break
                        else:
                            logging.error(f"获取上传URL失败，原因: {result.get('msg')}")
                    else:
                        logging.error(f"获取上传URL失败，状态码: {response.status_code}")
                except Exception as err:
                    logging.error(f"获取上传URL异常: {err}")
                
                if attempt < max_retries - 1:
                    logging.info(f"第{attempt + 1}次获取上传URL失败，等待{retry_interval}秒后重试...")
                    time.sleep(retry_interval)
            
            if not batch_id:
                logging.error(f"批次{batch_idx}获取上传URL失败，跳过该批次")
                continue
            
            batch_ids.append(batch_id)
            batch_id_path = os.path.join(log_dir, f'batch_id_{batch_idx}.txt')
            with open(batch_id_path, 'w', encoding='utf-8') as f:
                f.write(str(batch_id))
            
            # 处理文件上传
            successful_uploads = []
            failed_uploads = []
            urls = result["data"]["file_urls"]
            
            for i, url_info in enumerate(urls):
                file_path = batch_files[i]
                file_name = os.path.basename(file_path)
                
                for attempt in range(max_retries):
                    try:
                        with open(file_path, "rb") as f:
                            res_upload = requests.put(url_info, data=f)
                        
                        if res_upload.status_code == 200:
                            successful_uploads.append(file_name)
                            logging.info(f"{file_name} 上传成功")
                            break
                        else:
                            logging.warning(f"{file_name} 第{attempt + 1}次上传失败，状态码: {res_upload.status_code}")
                    except Exception as e:
                        logging.warning(f"{file_name} 第{attempt + 1}次上传异常: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_interval)
                else:
                    failed_uploads.append(file_name)
                    logging.error(f"{file_name} 上传失败，已达最大重试次数")
            
            # 记录上传结果
            logging.info(f"批次 {batch_idx}: 成功上传 {len(successful_uploads)}/{len(urls)} 个文件")
            
            if failed_uploads:
                # 将失败文件写入重试队列
                with open(retry_log, 'a') as f:
                    for file_name in failed_uploads:
                        f.write(f"{file_name}: {os.path.join(base_folder, file_name)}\n")
                
                logging.warning(f"批次 {batch_idx}: 以下文件上传失败: {', '.join(failed_uploads)}")
        
        # 如果存在重试日志，询问是否重试失败的上传
        if os.path.exists(retry_log) and os.path.getsize(retry_log) > 0:
            retry_choice = input("\n存在上传失败的文件，是否尝试重新上传？(yes/no): ").lower().strip()
            if retry_choice == 'yes':
                with open(retry_log, 'r') as f:
                    retry_lines = f.readlines()
                
                retry_files = []
                for line in retry_lines:
                    parts = line.strip().split(': ')
                    if len(parts) >= 2:
                        retry_files.append(parts[1])
                
                if retry_files:
                    logging.info(f"开始重试 {len(retry_files)} 个失败的上传文件...")
                    # 将重试文件作为新批次上传
                    retry_batch_ids = upload(False, base_folder, max_retries, retry_interval)
                    batch_ids.extend(retry_batch_ids)
        
        return batch_ids
    else:
        return []


def download_and_extract_zip(zip_url: str, output_dir: str, file_name: str):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        local_zip_path = os.path.join(output_dir, file_name) + ".zip"
        
        # 下载 ZIP
        r = requests.get(zip_url, stream=True)
        if r.status_code == 200:
            with open(local_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{file_name}.zip 下载完成")
            
            # 解压
            with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"{file_name}.zip 已解压到 {output_dir}")
            
            # 可删除 ZIP（可选）
            os.remove(local_zip_path)
            print(f"删除临时 ZIP 文件 {file_name}.zip")
        else:
            print(f"下载失败，状态码: {r.status_code}")
    except Exception as e:
        print(f"下载或解压出错: {e}")


def get(batch_id: str, processed_files=None, failed_files=None):
    """
    只处理未处理过的文件，已done的文件只处理一次，失败的文件可单独记录。
    :param batch_id: 批次id
    :param processed_files: 已经处理过的文件名集合
    :param failed_files: 失败的文件名集合
    """
    if processed_files is None:
        processed_files = set()
    if failed_files is None:
        failed_files = set()
    url_get = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    try:
        res = requests.get(url_get, headers=header)
        if res.status_code != 200:
            print(f"请求失败，状态码: {res.status_code}, 响应内容: {res.text}")
            return False, processed_files, failed_files
        try:
            res_json = res.json()
        except Exception as e:
            print(f"解析JSON失败: {e}\n原始响应内容: {res.text}")
            return False, processed_files, failed_files
        data_json = res_json.get("data", {})

        if "extract_result" in data_json and isinstance(data_json["extract_result"], list):
            all_done = True
            only_failed = True
            for result in data_json["extract_result"]:
                file_name = result["file_name"]
                state = result.get("state", "unknown")
                if file_name in processed_files:
                    continue  # 已处理过的文件不再打印和处理
                if state == "done":
                    only_failed = False
                    zip_url = result.get("full_zip_url")
                    if zip_url:
                        output_dir = f"./processed/{base_folder}/{file_name}/"
                        download_and_extract_zip(zip_url, output_dir, file_name)
                        processed_files.add(file_name)
                    else:
                        print(f"{file_name} 无 zip_url")
                elif state == "failed":
                    failed_files.add(file_name)
                    all_done = False
                else:
                    only_failed = False
                    all_done = False
                # 只打印未处理过的文件状态
                print(f"文件：{file_name} 状态: {state}")
            # 记录failed_files到log
            if failed_files:
                log_dir = os.path.join('log', os.path.basename(base_folder))
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, 'upload_fail.log')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n[失败文件] {sorted(list(failed_files))}\n")
            if all_done:
                print("所有文件处理完成")
                return True, processed_files, failed_files
            elif only_failed and failed_files:
                print("仅剩失败文件，终止轮询。")
                return True, processed_files, failed_files
            else:
                return False, processed_files, failed_files
        else:
            print("extract_result 为空或无效")
            return False, processed_files, failed_files
    except Exception as e:
        print(f"请求或处理结果时发生异常: {e}")
        return False, processed_files, failed_files


# 获取批量处理结果并轮询直到任务完成
has_upload = input("是否已上传文件？(yes/no): ").strip().lower() == "yes"
base_folder = input("请输入要上传的文件夹路径: ")
batch_ids = upload(has_upload, base_folder)
# batch_ids = ["102ffd1d-f56e-45ae-a6dc-9bb0240b384f"]  # 仅测试用

max_retry = 20
interval = 30
processed_files = set()
failed_files = set()
for batch_id in batch_ids:
    print(f"开始处理 batch_id: {batch_id}")
    for attempt in range(max_retry):
        finished, processed_files, failed_files = get(batch_id, processed_files, failed_files)
        if finished:
            print(f"batch_id: {batch_id} 所有文件处理完成，退出轮询。")
            break
        else:
            print(f"batch_id: {batch_id} 第{attempt+1}次轮询，尚有未完成文件，等待{interval}秒后重试...")
            time.sleep(interval)
    else:
        print(f"batch_id: {batch_id} 轮询已达最大次数，部分文件可能未处理完成。")