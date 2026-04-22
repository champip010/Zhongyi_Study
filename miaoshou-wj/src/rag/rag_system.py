# src/advanced_rag_system.py
# 设置环境变量确保离线模式
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import numpy as np
import torch
import json
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from sentence_transformers.util import cos_sim
from QdrantManager import QdrantManager
from vllm import LLM, SamplingParams
from sentence_transformers import CrossEncoder



# test
class MedicalRAG:
    
    def _configure_gpu_for_llm(self) -> Dict[str, Any]:
        """
        检测可使用显存>=15GB的GPU并设置可见性
        返回GPU配置信息
        """
        print("🔍 检测可使用显存>=15GB的GPU...")
        import subprocess
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，使用CPU模式")
            return {'llm_gpus': 0, 'memory_util': 0.0}
        
        # 检测符合条件的GPU（检查可用显存）
        try:
            suitable_gpus = []
            # 使用nvidia-smi获取GPU使用情况
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_id = int(parts[0])
                            memory_used = int(parts[1])
                            memory_total = int(parts[2])
                            free_memory = memory_total - memory_used
                            free_memory_gb = free_memory / 1024
                            
                            if free_memory_gb >= 16:
                                suitable_gpus.append(gpu_id)
                                print(f"✅ GPU {gpu_id}: 总显存{memory_total/1024**3:.1f}GB, 可用显存{free_memory_gb:.1f}GB")
        except Exception as e:
            print(f"⚠️ GPU 检测失败: {e}")
            return {'llm_gpus': 0, 'memory_util': 0.0}
    
        if not suitable_gpus:
            print("⚠️ 没有可用显存>=15GB的GPU，使用CPU模式")
            return {'llm_gpus': 0, 'memory_util': 0.0}
        
        # 限制GPU数量，避免tensor_parallel_size过大
        max_gpus = min(len(suitable_gpus), 4)  # 最多使用4个GPU
        selected_gpus = suitable_gpus[:max_gpus]
        
        # 设置CUDA_VISIBLE_DEVICES
        visible_devices = ','.join(map(str, selected_gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        
        # 配置LLM参数
        llm_gpus = max_gpus
        memory_util = 0.4 if llm_gpus >= 2 else 0.5
        
        print(f"🎯 设置CUDA_VISIBLE_DEVICES={visible_devices}")
        print(f"🎯 LLM使用{llm_gpus}个GPU，内存利用率{memory_util}")
        print(f"🎯 注意：QdrantManager将只能访问这些可见的GPU")
        
        return {
            'llm_gpus': llm_gpus,
            'memory_util': memory_util,
            'visible_gpus': selected_gpus
        }

    
    def __init__(
            self, 
            llm_model="Qwen/Qwen2-VL-7B-Instruct",  
            reset_qdrant: bool = False, 
            update_qdrant_knowledge_connection: bool = False,
            embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
            rerank_model_name: str = "BAAI/bge-reranker-large",
            chat: bool = False
        ):
         
        # GPU配置
        self.gpu_config = self._configure_gpu_for_llm()
        
        # 初始化配置
        self.config = {
        'llm_model': llm_model,
        'chunk_size': 600,
        'chunk_overlap': 150,
        "reset_qdrant" : reset_qdrant,  # 是否重置qdrant数据库
        "update_qdrant_knowledge_connection": update_qdrant_knowledge_connection,  # 是否更新qdrant知识库
        }
        
        self.llm = None
        # 只有search和chat的时候加载llm
        if chat:
            print("🔧 加载LLM模型...")
            self.llm = LLM(
                model=llm_model,
                tensor_parallel_size=self.gpu_config['llm_gpus'],
                max_model_len=4096,
                gpu_memory_utilization=self.gpu_config['memory_util'],
                trust_remote_code=True
            )
        


        # 初始化qdrant向量数据库（内置embedding模型）
        print("初始化Qdrant向量数据库...")
        
        self.qdrant = QdrantManager(
            embedding_model_name=embedding_model_name
        )
        
        # 使用中文交叉编码器
        self.reranker = CrossEncoder(rerank_model_name, max_length=512, device=self.qdrant.embedding_device)

        # HyDE的采样参数
        self.hyde_params = SamplingParams(temperature=0.7, max_tokens=300, stop=["<|im_end|>"])

        # 保留构建时需要的参数，但检索时优先使用QdrantManager
        self.documents = []
        self.embeddings = None

        # 重置qdrant数据库
        if self.config['reset_qdrant']:
            print("⚠️ 重置 Qdrant 集合...")
            try:
                self.qdrant.reset_collection()
                self.build_index()
            except Exception as e:
                print(f"重置集合失败: {e}")

        # 检查数据库状态
        if self.qdrant.count() > 0:
            print(f"qdrant数据库已存在，包含 {self.qdrant.count()} 条文档。")
            if self.config["update_qdrant_knowledge_connection"]:
                print("更新qdrant知识库连接...")
                self._build_knowledge_connections()
        else:
            print("qdrant数据库为空，需要重建embedding和入库。")
            self.build_index()
    

    def build_index(self, data_paths='../data/processed_md/', sample_limit: int = 0, file_limit: int = 0):
        print("🔬 构建先进语义索引(可限量/可重置/可限文件数)...")
        from data_processor import SimpleMedicalProcessor
        # 加载文档
        processor = SimpleMedicalProcessor()
        documents = processor.load_documents(data_paths, file_limit=file_limit)
        chunks = processor.smart_chunk_documents(documents)

        # 仅取少量样本以便快速测试
        if isinstance(sample_limit, int) and sample_limit > 0:
            chunks = chunks[:sample_limit]
            print(f"🧪 仅使用前 {len(chunks)} 个chunks进行测试构建")

        # 可选：清空并重置 Qdrant 集合


        # 语义分块 + 上下文增强
        enhanced_chunks = self._semantic_chunking_with_context(chunks)
        texts = [doc.page_content for doc in enhanced_chunks]

        print("🧬 生成高质量embeddings并写入qdrant...")
        embeddings = self.qdrant.encode_texts(texts, batch_size=16, convert_to_tensor=False)
        self.embeddings = embeddings
        self.documents = enhanced_chunks

        print("🧬 写入qdrant数据库...")
        # 先写入初始版本（connections为空）

        self.qdrant.add_documents(
            docs=self.documents,
            embeddings=self.embeddings,
        )
        print(f"已写入{len(self.documents)}个文档到qdrant。")

        # 如果文档较少，构建知识连接
        if len(self.documents) < 20000 or self.config["update_qdrant_knowledge_connection"] or self.config["reset_qdrant"]:
            print("🕸️ 构建知识连接")
            self._build_knowledge_connections()
            
        else:
            print(f"🕸️ 跳过知识连接构建（文档数量太大: {len(self.documents)}）")
            for doc in self.documents:
                doc.metadata['connections'] = []
        
        try:
            collection_info = self.client.get_collection(collection_name="hybrid-search")
            if collection_info.points_count == 0:
                print("混合搜索集合为空，正在创建...")
                self.create_hybrid_search_collection()
        except Exception:
            print("混合搜索集合不存在，正在创建...")
            self.create_hybrid_search_collection()

        print(f"✅ 索引构建完成: {len(self.documents)} 个增强文档块")


    def _semantic_chunking_with_context(self, documents: List[Document]) -> List[Document]:
        """语义分块 + 上下文增强（使用 jieba 进行中文分句）"""

        enhanced_chunks = []

        for doc in tqdm(documents, desc="分段/上下文增强", unit="doc"):
            # 使用 jieba 进行中文分句
            paragraphs = self._chinese_sentence_split(doc.page_content)
            
            if not paragraphs:
                continue

            # 一次性批量计算 embedding（减少模型调用次数）
            para_embeddings = self.qdrant.encode_texts(paragraphs, batch_size=16, convert_to_tensor=False)

            current_chunk = ""
            chunk_embeddings = []

            for i, (para, para_emb) in enumerate(zip(paragraphs, para_embeddings)):
                if not chunk_embeddings or len(current_chunk) < 300:
                    current_chunk += para + '\n\n'
                    chunk_embeddings.append(para_emb)
                else:
                    # 检查语义相似度
                    avg_chunk_emb = np.mean(chunk_embeddings, axis=0)
                    similarity = cos_sim(
                        torch.tensor(para_emb), torch.tensor(avg_chunk_emb)
                    ).item()

                    if similarity > 0.8 and len(current_chunk) < 800:
                        current_chunk += para + '\n\n'
                        chunk_embeddings.append(para_emb)
                    else:
                        # 保存当前 chunk
                        if current_chunk.strip():
                            enhanced_chunk = self._add_context(current_chunk, doc, i)
                            enhanced_chunks.append(enhanced_chunk)

                        current_chunk = para + '\n\n'
                        chunk_embeddings = [para_emb]

            # 保存最后一个 chunk
            if current_chunk.strip():
                enhanced_chunk = self._add_context(current_chunk, doc, len(paragraphs))
                enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _chinese_sentence_split(self, text: str) -> List[str]:
        """使用 jieba 进行中文分句"""
        import jieba
        
        # 中文分句标点符号
        sentence_endings = ['。', '！', '？', '；', '\n\n', '\n']
        
        # 使用 jieba 进行分词
        words = list(jieba.cut(text))
        
        sentences = []
        current_sentence = ""
        
        for word in words:
            current_sentence += word
            
            # 检查是否遇到句子结束符号
            if any(ending in current_sentence for ending in sentence_endings):
                # 找到最后一个句子结束符号的位置
                for ending in sentence_endings:
                    if ending in current_sentence:
                        # 按结束符号分割
                        parts = current_sentence.split(ending)
                        for i, part in enumerate(parts[:-1]):
                            if part.strip():
                                sentences.append(part.strip() + ending)
                        current_sentence = parts[-1]
                        break
        
        # 添加最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 智能合并短句子，而不是直接舍弃
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            
            # 如果当前句子太短，尝试与后续句子合并
            if len(current.strip()) < 20 and i + 1 < len(sentences):
                # 合并当前句子和下一个句子
                merged = current + " " + sentences[i + 1]
                merged_sentences.append(merged)
                i += 2  # 跳过下一个句子
            else:
                # 句子长度足够，直接添加
                merged_sentences.append(current)
                i += 1
        
        return merged_sentences
    
    def _add_context(self, chunk_content: str, original_doc: Document, position: int) -> Document:
        """为chunk添加上下文信息（保持原文本不变，增强信息放在metadata中）"""
        
        # 提取标题信息
        doc_content = original_doc.page_content
        titles = []
        for line in doc_content.split('\n'):
            if line.startswith('#'):
                titles.append(line.strip())
        
        # 保持原文本内容不变
        enhanced_content = chunk_content
        
        # 增强metadata，包含所有上下文信息
        enhanced_metadata = original_doc.metadata.copy()
        enhanced_metadata.update({
            'chunk_position': position,
            'has_titles': len(titles) > 0,
            'content_type': self._classify_content_type(chunk_content),
            'medical_entities': ",".join(self._extract_medical_entities(chunk_content)),
            'document_titles': titles,  # 添加文档标题列表
            'chunk_length': len(chunk_content),  # 添加chunk长度
            'enhanced_at': 'semantic_chunking'  # 标记增强方式
        })
        
        return Document(
            page_content=enhanced_content,
            metadata=enhanced_metadata
        )
    
    def _classify_content_type(self, content: str) -> str:
        """分类内容类型（优化版本）"""
        # 使用集合操作提高效率
        content_lower = content.lower()
        
        # 预定义关键词集合
        diagnosis_keywords = {'诊断', '标准', '症状', '表现', '体征', '临床表现'}
        treatment_keywords = {'治疗', '干预', '疗法', '方法', '手术', '康复'}
        medication_keywords = {'药物', '剂量', '副作用', '用药', '处方', '药理'}
        assessment_keywords = {'评估', '量表', '评分', '检查', '检测', '化验'}
        
        # 使用集合交集快速判断
        if any(keyword in content_lower for keyword in diagnosis_keywords):
            return 'diagnosis'
        elif any(keyword in content_lower for keyword in treatment_keywords):
            return 'treatment'  
        elif any(keyword in content_lower for keyword in medication_keywords):
            return 'medication'
        elif any(keyword in content_lower for keyword in assessment_keywords):
            return 'assessment'
        else:
            return 'general'

    def _extract_medical_entities(self, content: str) -> List[str]:
        """提取医学实体（优化版本）"""
        # 使用集合提高查找效率
        medical_keywords = {
            # 疾病类
            '糖尿病', '高血压', '冠心病', '心肌梗死', '脑卒中', '肺炎', '哮喘', 
            '肝炎', '肾炎', '胃炎', '肠炎', '失眠', '帕金森病',
            
            # 症状类
            '发热', '疼痛', '咳嗽', '呼吸困难', '头痛', '恶心', '多梦', '早醒',
            
            # 治疗类
            '手术', '药物治疗', '化疗', '放疗', '康复', '镇静催眠药',
            
            # 检查类
            'CT', 'MRI', 'X光', '超声', '心电图', '血检', 'PSQI',
            
            # 中医类
            '肝气郁结', '心脾两虚', '心肾不交', '阴虚火旺', '戒断综合征'
        }
        
        # 使用集合交集快速查找
        found_terms = [term for term in medical_keywords if term in content]
        return found_terms

    def hyde_enhanced_query(self, query: str) -> str:
        """HyDE: 生成假设文档来增强查询"""
        
        hyde_prompt = f"""作为资深医学专家，针对问题"{query}"，请生成一个简洁的假设性医学答案，包含可能的关键医学术语和概念。"""
        
        try:
            outputs = self.llm.generate([hyde_prompt], self.hyde_params)
            hypothetical_doc = outputs[0].outputs[0].text.strip()
            
            # 将原查询和假设文档结合
            enhanced_query = f"{query} {hypothetical_doc}"
            return enhanced_query
            
        except Exception as e:
            print(f"⚠️ HyDE生成失败，使用原查询: {e}")
            return query

     
    def _build_knowledge_connections(self, top_k: int = 10):
        """构建知识连接（使用 qdrant 近邻搜索替代 O(n²)）
        - 优化：直接使用QdrantManager，不依赖预加载的文档
        """

        doc_count = self.qdrant.count()
        print(f"🕸️ 开始构建知识连接，共 {doc_count} 个文档...")
        
        # 分批处理避免内存问题
        batch_size = 500
        for start_idx in range(0, doc_count, batch_size):
            end_idx = min(start_idx + batch_size, doc_count)
            doc_ids = list(range(start_idx, end_idx))
            
            # 使用Qdrant批量操作
            self.qdrant.batch_update_connections(doc_ids)
            print(f"🕸️ 已处理 {end_idx}/{doc_count} 个文档的连接...")
        
        print("✅ 知识连接构建完成！")

    def multi_hop_retrieval(self, query: str, max_hops: int = 2, target_k: int = 5) -> List[Document]:
        """多跳检索：基于初始检索结果进行扩展检索
        - target_k: 目标返回的文档数量，由search方法的k参数决定
        - 智能选择：如果预加载了文档则使用内存版本，否则使用QdrantManager
        """
        
        # 第1跳：基础检索 - 使用HyDE增强查询
        enhanced_query = self.hyde_enhanced_query(query)
        initial_docs = self._dense_retrieval(enhanced_query, top_k=target_k)
        
        if max_hops <= 1:
            return initial_docs[:target_k]
        
        # 第2跳：基于连接扩展 - 智能选择数据源
        expanded_docs = []
        seen_doc_ids = set()
        seen_contents = set()  # 添加内容去重
        
        for doc in initial_docs:
            # 从文档元数据中获取连接信息
            connections = doc.metadata.get('connections', [])
            connections = json.loads(connections) if isinstance(connections, str) else connections
            
            for conn in connections:  # 每个文档最多扩展5个连接
                doc_id = conn['doc_id']
                similarity = conn['similarity']
                # 更严格的相似度过滤：避免检索到几乎完全相同的文档
                if doc_id not in seen_doc_ids and similarity < 0.95:  # 从0.98改为0.85
                    # 智能选择：如果预加载了文档且doc_id在范围内，使用内存版本
                    if self.documents and 0 <= doc_id < len(self.documents):
                        connected_doc = self.documents[doc_id]
                        # 内容级别去重
                        content_key = connected_doc.page_content[:100]
                        if content_key not in seen_contents:
                            expanded_docs.append(connected_doc)
                            seen_doc_ids.add(doc_id)
                            seen_contents.add(content_key)
                    else:
                        # 否则使用QdrantManager获取
                        connected_doc = self.qdrant.get_document(doc_id)
                        if connected_doc:
                            # 内容级别去重
                            content_key = connected_doc.page_content[:100]
                            if content_key not in seen_contents:
                                expanded_docs.append(connected_doc)
                                seen_doc_ids.add(doc_id)
                                seen_contents.add(content_key)
        
        # 重新排序所有文档
        all_docs = initial_docs + expanded_docs
        
        # 最终去重：使用智能内容相似度检测
        final_docs = []
        final_contents = []
        
        for doc in all_docs:
            # 检查是否与已有文档内容过于相似
            is_duplicate = False
            for existing_content in final_contents:
                overlap_ratio = self._calculate_content_overlap(doc.page_content[:300], existing_content[:300])
                if overlap_ratio > 0.7:  # 如果重叠度超过70%，认为是重复
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_docs.append(doc)
                final_contents.append(doc.page_content[:300])
                if len(final_docs) >= target_k:
                    break
        
        # 如果去重后文档不足，从原始结果中补充（但跳过明显重复的）
        if len(final_docs) < target_k:
            for doc in all_docs:
                if doc not in final_docs:
                    # 再次检查是否与最终结果重复
                    is_duplicate = False
                    for existing_content in final_contents:
                        overlap_ratio = self._calculate_content_overlap(doc.page_content[:300], existing_content[:300])
                        if overlap_ratio > 0.6:  # 稍微放宽阈值
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        final_docs.append(doc)
                        final_contents.append(doc.page_content[:300])
                        if len(final_docs) >= target_k:
                            break
        
        reranked_docs = self._rerank_with_cross_encoder(query, final_docs)
        
        return reranked_docs[:target_k]  # 返回目标数量的文档
    
    def qdrant_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        results = self.qdrant.query_text(query, n_results=top_k)
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=doc, metadata=meta))
        return docs

    def _dense_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """密集检索，自动去重并确保返回足够的文档"""
        # 获取更多候选文档用于去重
        candidates = self.qdrant.search_text(query, top_k=top_k * 3)  # 增加候选数量
        
        # 基于内容相似度去重（更严格的去重）
        unique_docs = []
        seen_contents = set()
        
        for doc in candidates:
            # 使用更长的内容片段进行去重，提高准确性
            content_key = doc.page_content[:200]  # 从100增加到200
            
            # 检查是否与已有文档内容过于相似
            is_duplicate = False
            for existing_content in seen_contents:
                # 计算内容重叠度
                overlap_ratio = self._calculate_content_overlap(content_key, existing_content)
                if overlap_ratio > 0.8:  # 如果重叠度超过80%，认为是重复
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.add(content_key)
                
                # 如果已经收集到足够的文档，提前结束
                if len(unique_docs) >= top_k:
                    break
        
        return unique_docs[:top_k]
    
    def _calculate_content_overlap(self, text1: str, text2: str) -> float:
        """计算两段文本的内容重叠度"""
        if not text1 or not text2:
            return 0.0
        
        # 将文本分割成字符集合
        chars1 = set(text1)
        chars2 = set(text2)
        
        # 计算Jaccard相似度
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _rerank_with_cross_encoder(self, query: str, docs: List[Document]) -> List[Document]:
        """使用交叉编码器重排序"""
        try:
            
            # 准备query-doc对
            pairs = [(query, doc.page_content[:500]) for doc in docs]  # 截断长文本
            
            # 计算重排序分数
            scores = self.reranker.predict(pairs)
            
            # 按分数排序
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            print(f"⚠️ 重排序失败，使用原顺序: {e}")
            return docs

    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """混合检索：结合语义检索和BM25检索"""
        try:
            # 使用新的高级混合搜索方法
            docs = self.qdrant.advanced_hybrid_search(query, top_k=top_k)
            if docs:
                print(f"✅ 混合检索成功，找到 {len(docs)} 个文档")
                return docs
        except Exception as e:
            print(f"⚠️ 混合检索失败: {e}")


    def search(self, query: str, k: int = 5, use_hybrid_retrieval: bool = False) -> List[Document]:
        """对外暴露：根据查询返回包含元数据的文档列表，兼容 qa_base 调用。
        
        - 文档数量由k参数决定，不再使用固定值
        - 兼容字段：确保 metadata 包含 'file_name'（qa_base 展示所需）。
        """
        docs: List[Document] = []


        if use_hybrid_retrieval:
            try:
                docs = self.hybrid_retrieval(query, top_k=k)
                return docs
            except Exception as e:
                print(f"⚠️ 混合检索失败: {e}")
                return docs

        if self.llm == None:
            print("加载LLM...")
            try:
                # vLLM会自动使用可见的GPU，我们只需要设置数量和内存利用率
                self.llm = LLM(
                    model=self.config['llm_model'],
                    tensor_parallel_size=self.gpu_config['llm_gpus'],
                    max_model_len=4096,
                    gpu_memory_utilization=self.gpu_config['memory_util'],
                    trust_remote_code=True
                )
                print(f"✅ LLM模型加载成功，使用{self.gpu_config['llm_gpus']}个GPU")
            except Exception as e:
                print(f"❌ LLM模型加载失败: {e}")
                print("⚠️ 将跳过LLM加载")
                self.llm = None

        # 回退：多跳检索    
        if not docs:
            try:
                docs = self.multi_hop_retrieval(query, max_hops=2, target_k=k)
            except Exception as e:
                print(f"⚠️ 多跳检索失败: {e}")
                docs = []

        # 回退：基础 Qdrant 检索
        if not docs:
            try:
                docs = self.qdrant_retrieval(query, top_k=k)
            except Exception as e:
                print(f"⚠️ Qdrant检索失败: {e}")
                docs = []

        # 限制数量到k
        docs = docs[:k]

        # 兼容 qa_base：补齐 'file_name'
        for doc in docs:
            meta = doc.metadata or {}
            if 'file_name' not in meta:
                if 'source' in meta:
                    meta['file_name'] = meta['source']
                elif 'relative_path' in meta:
                    meta['file_name'] = meta['relative_path'].split('/')[-1]
                else:
                    meta['file_name'] = 'unknown.md'
            doc.metadata = meta

        return docs
