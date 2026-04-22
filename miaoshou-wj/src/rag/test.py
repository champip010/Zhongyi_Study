import os
import time
import json
from datetime import datetime

# 确保离线/显卡设置
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from rag_system import MedicalRAG
from QdrantManager import QdrantManager
from data_processor import SimpleMedicalProcessor

class TestResultRecorder:
    """测试结果记录器"""
    
    def __init__(self):
        self.results = {
            "test_start_time": None,
            "test_end_time": None,
            "system_build": {},
            "search_performance": {},
            "qdrant_direct": {},
            "memory_usage": {},
            "summary": {}
        }
        self.log_messages = []
    
    def log(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_messages.append(log_entry)
    
    def record_system_build(self, build_time, success, error=None):
        """记录系统构建结果"""
        self.results["system_build"] = {
            "build_time_seconds": build_time,
            "build_time_minutes": build_time / 60,
            "success": success,
            "error": str(error) if error else None
        }
    
    def record_search_performance(self, queries, search_times, hybrid_times, doc_counts):
        """记录搜索性能结果"""
        self.results["search_performance"] = {
            "total_queries": len(queries),
            "successful_searches": len(search_times),
            "search_times": search_times,
            "hybrid_times": hybrid_times,
            "doc_counts": doc_counts,
            "avg_search_time": sum(search_times) / len(search_times) if search_times else 0,
            "avg_hybrid_time": sum(hybrid_times) / len(hybrid_times) if hybrid_times else 0
        }
    
    def record_qdrant_direct(self, init_time, query_time, success, error=None):
        """记录QdrantManager直接测试结果"""
        self.results["qdrant_direct"] = {
            "init_time": init_time,
            "query_time": query_time,
            "success": success,
            "error": str(error) if error else None
        }
    
    def record_memory_usage(self, system_memory, gpu_memory):
        """记录内存使用情况"""
        self.results["memory_usage"] = {
            "system_memory": system_memory,
            "gpu_memory": gpu_memory
        }
    
    def record_summary(self, doc_count, total_test_time):
        """记录测试总结"""
        self.results["summary"] = {
            "final_doc_count": doc_count,
            "total_test_time_seconds": total_test_time,
            "total_test_time_minutes": total_test_time / 60
        }
    
    def save_results(self, filename_prefix="test_results"):
        """保存测试结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON结果
        json_filename = f"{filename_prefix}_{timestamp}.json"
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"✅ 测试结果已保存到: {json_filename}")
        except Exception as e:
            print(f"❌ 保存JSON结果失败: {e}")
        
        # 保存详细日志
        log_filename = f"{filename_prefix}_{timestamp}.log"
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"RAG系统测试日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                for log_entry in self.log_messages:
                    f.write(log_entry + "\n")
            print(f"✅ 详细日志已保存到: {log_filename}")
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")
        
        return json_filename, log_filename

def print_separator(title):
    """打印分隔线"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def test_system_build(recorder):
    """测试系统构建性能"""
    print_separator("系统构建性能测试")
    
    start_time = time.time()
    recorder.log(f"🕐 开始构建时间: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # 仅用于构建索引，不加载大模型推理
        rag = MedicalRAG(reset_qdrant=True)
        
        build_time = time.time() - start_time
        recorder.log(f"✅ 系统构建完成！")
        recorder.log(f"⏱️  总构建时间: {build_time:.2f}秒 ({build_time/60:.2f}分钟)")
        
        recorder.record_system_build(build_time, True)
        return rag
    except Exception as e:
        build_time = time.time() - start_time
        recorder.log(f"❌ 系统构建失败: {e}")
        recorder.record_system_build(build_time, False, e)
        return None

def test_search_performance(rag, test_queries, recorder):
    """测试搜索性能"""
    print_separator("搜索性能测试")
    
    if not rag:
        recorder.log("❌ RAG系统未初始化，跳过搜索测试")
        return
    
    search_times = []
    hybrid_times = []
    doc_counts = []
    
    for i, query in enumerate(test_queries, 1):
        recorder.log(f"\n🔍 测试查询 {i}/{len(test_queries)}: {query}")
        
        # 测试基础搜索
        start_time = time.time()
        try:
            docs = rag.search(query=query, k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
            doc_counts.append(len(docs))
            
            recorder.log(f"✅ 基础搜索完成，耗时: {search_time:.3f}秒")
            recorder.log(f"📚 找到 {len(docs)} 个相关文档")
            
            # 显示前2个文档的摘要
            for j, doc in enumerate(docs[:2]):
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                recorder.log(f"   📄 文档{j+1}: {content_preview}")
                
        except Exception as e:
            recorder.log(f"❌ 搜索失败: {e}")
            search_times.append(0)
            doc_counts.append(0)
        
        # 测试混合搜索
        start_time = time.time()
        try:
            hybrid_docs = rag.search(query=query, k=5, use_hybrid_retrieval=True)
            hybrid_time = time.time() - start_time
            hybrid_times.append(hybrid_time)
            
            recorder.log(f"✅ 混合搜索完成，耗时: {hybrid_time:.3f}秒")
            recorder.log(f"📚 找到 {len(hybrid_docs)} 个相关文档")
            
        except Exception as e:
            recorder.log(f"❌ 混合搜索失败: {e}")
            hybrid_times.append(0)
    
    if search_times:
        successful_searches = len([t for t in search_times if t > 0])
        avg_search_time = sum(search_times) / successful_searches if successful_searches > 0 else 0
        recorder.log(f"\n📊 搜索性能统计:")
        recorder.log(f"   ✅ 成功搜索次数: {successful_searches}")
        recorder.log(f"   ⏱️  平均搜索时间: {avg_search_time:.3f}秒")
        recorder.log(f"   ⏱️  总搜索时间: {sum(search_times):.3f}秒")
    
    recorder.record_search_performance(test_queries, search_times, hybrid_times, doc_counts)

def test_qdrant_direct(recorder):
    """直接测试QdrantManager性能"""
    print_separator("QdrantManager直接测试")
    
    start_time = time.time()
    try:
        qdrant = QdrantManager()
        init_time = time.time() - start_time
        recorder.log(f"✅ QdrantManager初始化完成，耗时: {init_time:.3f}秒")
        
        # 测试直接查询
        test_query = "高血压的治疗方法"
        start_time = time.time()
        docs = qdrant.advanced_hybrid_search(query=test_query, top_k=5)
        query_time = time.time() - start_time
        
        recorder.log(f"✅ 直接查询完成，耗时: {query_time:.3f}秒")
        recorder.log(f"📚 找到 {len(docs)} 个相关文档")
        
        recorder.record_qdrant_direct(init_time, query_time, True)
        return qdrant
        
    except Exception as e:
        recorder.log(f"❌ QdrantManager测试失败: {e}")
        recorder.record_qdrant_direct(0, 0, False, e)
        return None

def test_memory_usage(recorder):
    """测试内存使用情况"""
    print_separator("内存使用测试")
    
    system_memory = {}
    gpu_memory = {}
    
    try:
        import psutil
        import torch
        
        # 系统内存
        memory = psutil.virtual_memory()
        system_memory = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "usage_percent": memory.percent
        }
        
        recorder.log(f"💾 系统内存: {system_memory['total_gb']:.1f}GB")
        recorder.log(f"💾 可用内存: {system_memory['available_gb']:.1f}GB")
        recorder.log(f"💾 内存使用率: {system_memory['usage_percent']:.1f}%")
        
        # GPU内存
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            recorder.log(f"🎮 检测到 {gpu_count} 个GPU设备")
            
            for i in range(gpu_count):
                try:
                    gpu_memory_props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    gpu_memory[f"gpu_{i}"] = {
                        "name": gpu_memory_props.name,
                        "allocated_gb": allocated,
                        "cached_gb": cached
                    }
                    
                    recorder.log(f"   GPU {i}: {gpu_memory_props.name}")
                    recorder.log(f"     已分配: {allocated:.2f}GB")
                    recorder.log(f"     已缓存: {cached:.2f}GB")
                except Exception as e:
                    recorder.log(f"   GPU {i}: 无法获取信息 - {e}")
                    gpu_memory[f"gpu_{i}"] = {"error": str(e)}
        else:
            recorder.log("⚠️ CUDA不可用")
            
    except ImportError:
        recorder.log("⚠️ 无法导入psutil，跳过内存测试")
    
    recorder.record_memory_usage(system_memory, gpu_memory)

def main():
    """主测试函数"""
    # 初始化结果记录器
    recorder = TestResultRecorder()
    recorder.results["test_start_time"] = datetime.now().isoformat()
    
    print_separator("RAG系统全面性能测试")
    recorder.log(f"🚀 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试用例
    test_queries = [
        "什么是糖尿病？",
        "高血压的治疗方法有哪些？",
        "失眠的症状和原因",
        "心脏病的预防措施",
        "抑郁症的诊断标准",
        "慢性胃炎如何治疗？",
        "哮喘的急救方法",
        "骨质疏松的预防",
        "脑卒中的早期症状",
        "癌症的早期筛查"
    ]
    
    # 1. 系统构建测试
    rag = test_system_build(recorder)

    # 2. 搜索性能测试
    test_search_performance(rag, test_queries, recorder)
    
    # 3. QdrantManager直接测试
    qdrant = test_qdrant_direct(recorder)
    
    # 4. 内存使用测试
    test_memory_usage(recorder)
    
    # 5. 最终统计
    print_separator("测试完成总结")
    recorder.log(f"🏁 测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_test_time = time.time() - time.mktime(datetime.fromisoformat(recorder.results["test_start_time"]).timetuple())
    
    if rag:
        try:
            doc_count = rag.qdrant.count()
            recorder.log(f"📊 数据库状态: {doc_count} 个文档")
            recorder.record_summary(doc_count, total_test_time)
        except:
            recorder.log("📊 数据库状态: 无法获取")
            recorder.record_summary(0, total_test_time)
    
    recorder.results["test_end_time"] = datetime.now().isoformat()
    
    # 保存结果
    json_file, log_file = recorder.save_results()
    
    print(f"\n🎉 所有测试完成！")
    print(f"📁 结果文件: {json_file}")
    print(f"📁 日志文件: {log_file}")

if __name__ == "__main__":
    # main()
    rag = MedicalRAG(reset_qdrant=False)
    docs = rag.search(query="什么是糖尿病？", k=5)
    for doc in docs:
        print(doc)
    docs = rag.search(query="什么是糖尿病？", k=5,use_hybrid_retrieval=True)
    for doc in docs:
        print(doc)
    # processor = SimpleMedicalProcessor()
    # documents = processor.load_documents("/home/wangjun/study/rag/wj-miaoshou/miaoshou-wj-rag/src/data/p-md-test")
    # for doc in documents:
    #     print(doc.metadata)
    #     print(doc.page_content[:200])
    #     print("-"*100+"\n")
    # chunks = processor.smart_chunk_documents(documents)
    
    # for chunk in chunks:
    #     print(chunk.metadata)
    #     print(chunk.page_content)
    #     print(len(chunk.page_content))
    #     print("-"*100+"\n")
    # print(f"chunks: {len(chunks)}")
    # print(f"chunks length: {sum([len(chunk.page_content) for chunk in chunks])/len(chunks)}")

    # enhanced_chunks = rag._semantic_chunking_with_context(chunks)
    # for chunk in enhanced_chunks[:10]:
    #     print(chunk.metadata)
    #     print(chunk.page_content[:200])
    #     print(len(chunk.page_content))
    #     print("-"*100+"\n")
    # print(f"enhanced_chunks: {len(enhanced_chunks)}")
    # print(f"enhanced_chunks length: {sum([len(chunk.page_content) for chunk in enhanced_chunks])/len(enhanced_chunks)}")