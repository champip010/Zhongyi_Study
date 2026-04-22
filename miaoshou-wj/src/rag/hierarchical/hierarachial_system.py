# src/advanced_rag_system.py
import os
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
from hierarchical_index import HierarchicalIndexManager

# 设置环境变量确保离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

# test
class MedicalRAG:
    def __init__(self, llm_model="Qwen/Qwen2-VL-7B-Instruct", load_llm: bool = True):
        
        # 初始化配置
        self.config = {
        'chunk_size': 600,
        'chunk_overlap': 150,
        "reset_qdrant" : False,  # 是否重置qdrant数据库
        "update_qdrant_knowledge_connection": False,  # 是否更新qdrant知识库
        }
        
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 可选加载 LLM（构建数据库模式可跳过）
        self.llm = None
        if load_llm:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6"
            print("加载RAG模型...")
            self.llm = LLM(
            model=llm_model,
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

        # 初始化qdrant向量数据库（内置embedding模型）
        print("初始化Qdrant向量数据库...")
        qw = "Qwen/Qwen3-Embedding-0.6B"
        bge = "BAAI/bge-large-zh-v1.5"
        self.qdrant = QdrantManager(
            embedding_model_name=bge,
            embedding_device="cuda:3"
        )
        
        # 初始化分层索引管理器
        self.hierarchical_index = HierarchicalIndexManager(self.qdrant)
        
        # HyDE和Self-RAG的采样参数
        self.hyde_params = SamplingParams(temperature=0.7, max_tokens=300, stop=["<|im_end|>"])
        self.generation_params = SamplingParams(temperature=0.3, max_tokens=800, stop=["<|im_end|>"])
        self.critique_params = SamplingParams(temperature=0.1, max_tokens=100, stop=["<|im_end|>"])
        
        self.documents = []
        self.embeddings = None

        if self.qdrant.count() > 0:
            print("qdrant数据库已存在，直接加载文档元数据，无需重建embedding。")
            self._load_documents_from_qdrant()
            if self.config["update_qdrant_knowledge_connection"]:
                print("更新qdrant知识库连接...")
                self._build_knowledge_connections()
        else:
            print("qdrant数据库为空，需要重建embedding和入库。")
    
    def _load_documents_from_qdrant(self):

        results = self.qdrant.get_all()
        self.embeddings = results["embeddings"]
        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        print(f"已从qdrant加载{len(self.documents)}条文档。")

    def build_index(self,data_paths='../data/processed_md/'):
        print("🔬 构建分层语义索引...")
        from data_processor import SimpleMedicalProcessor
        # 加载文档
        processor = SimpleMedicalProcessor()
        documents = processor.load_documents(data_paths)
        chunks = processor.smart_chunk_documents(documents)

        # 使用分层索引管理器构建索引
        print("🧬 构建分层索引...")
        index_stats = self.hierarchical_index.build_hierarchical_index(chunks)
        
        print(f"✅ 分层索引构建完成:")
        print(f"   - 文档级别: {index_stats['documents']} 个")
        print(f"   - 段落级别: {index_stats['paragraphs']} 个") 
        print(f"   - 句子级别: {index_stats['sentences']} 个")
        
        # 保存文档引用用于后续检索
        self.documents = chunks


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
                if not chunk_embeddings or len(current_chunk) < 400:
                    current_chunk += para + '\n\n'
                    chunk_embeddings.append(para_emb)
                else:
                    # 检查语义相似度
                    avg_chunk_emb = np.mean(chunk_embeddings, axis=0)
                    similarity = cos_sim(
                        torch.tensor(para_emb), torch.tensor(avg_chunk_emb)
                    ).item()

                    if similarity > 0.7 and len(current_chunk) < 800:
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
        """为chunk添加上下文信息"""
        
        # 提取标题信息
        doc_content = original_doc.page_content
        titles = []
        for line in doc_content.split('\n'):
            if line.startswith('#'):
                titles.append(line.strip())
        
        # 构建上下文增强的内容
        context_prefix = ""
        if titles:
            relevant_title = titles[-1] if titles else ""
            context_prefix = f"【文档标题】{relevant_title}\n\n"
        
        enhanced_content = context_prefix + chunk_content
        
        # 增强metadata
        enhanced_metadata = original_doc.metadata.copy()
        enhanced_metadata.update({
            'chunk_position': position,
            'has_titles': len(titles) > 0,
            'content_type': self._classify_content_type(chunk_content),
            'medical_entities': ",".join(self._extract_medical_entities(chunk_content))  # 这里做修改
        })
        
        return Document(
            page_content=enhanced_content,
            metadata=enhanced_metadata
        )
    
    def _classify_content_type(self, content: str) -> str:
        """分类内容类型"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['诊断', '标准', '症状', '表现']):
            return 'diagnosis'
        elif any(word in content_lower for word in ['治疗', '干预', '疗法', '方法']):
            return 'treatment'  
        elif any(word in content_lower for word in ['药物', '剂量', '副作用']):
            return 'medication'
        elif any(word in content_lower for word in ['评估', '量表', '评分']):
            return 'assessment'
        else:
            return 'general'
    
    # 在 advanced_rag_system.py 中只修改医学术语和prompt，保留所有其他先进技术

    def _extract_medical_entities(self, content: str) -> List[str]:
        """提取医学实体（扩展通用医学版）"""
        medical_terms = []
        
        # 扩展医学术语库（保留原有 + 新增通用）
        medical_keywords = [
            # 原有睡眠相关（保留）
            '失眠', '多梦', '早醒', '入睡困难', '睡眠质量', 
            'PSQI', '帕金森病', '镇静催眠药', '戒断综合征',
            '肝气郁结', '心脾两虚', '心肾不交', '阴虚火旺',
            
            # 新增通用医学术语
            '高血压', '糖尿病', '冠心病', '心肌梗死', '脑卒中',
            '肺炎', '哮喘', '肝炎', '肾炎', '胃炎', '肠炎',
            '发热', '疼痛', '咳嗽', '呼吸困难', '头痛', '恶心',
            '手术', '药物治疗', '化疗', '放疗', '康复',
            'CT', 'MRI', 'X光', '超声', '心电图', '血检'
        ]
        
        for term in medical_keywords:
            if term in content:
                medical_terms.append(term)
        
        return medical_terms

    def hyde_enhanced_query(self, query: str) -> str:
        """HyDE: 生成假设文档来增强查询"""
        
        hyde_prompt = f"""作为资深医学专家，针对问题"{query}"，请生成一个简洁的假设性医学答案，包含可能的关键医学术语和概念。

    问题：{query}

    假设答案："""
        
        try:
            outputs = self.llm.generate([hyde_prompt], self.hyde_params)
            hypothetical_doc = outputs[0].outputs[0].text.strip()
            
            # 将原查询和假设文档结合
            enhanced_query = f"{query} {hypothetical_doc}"
            return enhanced_query
            
        except Exception as e:
            print(f"⚠️ HyDE生成失败，使用原查询: {e}")
            return query

    def _assess_retrieval_need(self, query: str) -> bool:
        """评估是否需要检索"""
        
        assessment_prompt = f"""请判断回答以下医学问题是否需要参考专业文档资料。

    问题：{query}

    如果需要参考专业资料（如诊断标准、治疗指南、药物信息等），回答"需要"；
    如果是常识性问题可以直接回答，回答"不需要"。

    判断："""
        
        try:
            outputs = self.llm.generate([assessment_prompt], self.critique_params)
            assessment = outputs[0].outputs[0].text.strip()
            
            return "需要" in assessment
            
        except Exception as e:
            print(f"⚠️ 检索需求评估失败，默认使用检索: {e}")
            return True

    def _generate_direct_answer(self, query: str) -> str:
        """直接生成答案（无检索）"""
        
        direct_prompt = f"""作为资深医学专家，请回答以下问题：

    {query}

    回答："""
        
        try:
            outputs = self.llm.generate([direct_prompt], self.generation_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            return f"生成答案时出错：{str(e)}"

    def _generate_rag_answer(self, query: str, docs: List[Document]) -> str:
        """基于检索文档生成答案"""
        
        context = "\n\n".join([
            f"【参考资料{i+1}】{doc.page_content[:600]}" 
            for i, doc in enumerate(docs)
        ])
        
        rag_prompt = f"""作为资深医学专家，请基于以下参考资料回答问题。

    参考资料：
    {context}

    问题：{query}

    请提供准确、专业的回答："""
        
        try:
            outputs = self.llm.generate([rag_prompt], self.generation_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            return f"生成RAG答案时出错：{str(e)}"
     
    def _build_knowledge_connections(self, top_k: int = 10):
        """构建知识连接（使用 qdrant 近邻搜索替代 O(n²)）"""

        doc_count = self.qdrant.count()
        
        # 分批处理避免内存问题
        batch_size = 500
        for start_idx in range(0, doc_count, batch_size):
            end_idx = min(start_idx + batch_size, doc_count)
            doc_ids = list(range(start_idx, end_idx))
            
            # 使用Qdrant批量操作
            self.qdrant.batch_update_connections(doc_ids)
    
    def multi_hop_retrieval(self, query: str, max_hops: int = 2) -> List[Document]:
        """多跳检索：基于初始检索结果进行扩展检索"""
        
        # 第1跳：基础检索
        enhanced_query = self.hyde_enhanced_query(query)
        initial_docs = self._dense_retrieval(enhanced_query, top_k=3)
        
        if max_hops <= 1:
            return initial_docs
        
        # 第2跳：基于连接扩展
        expanded_docs = []
        seen_indices = set(self.documents.index(doc) for doc in initial_docs)
        
        for doc in initial_docs:
            connections = doc.metadata.get('connections', [])
            connections = json.loads(connections) if isinstance(connections, str) else connections
            
            for conn in connections[:2]:  # 每个文档最多扩展2个连接
                doc_id = conn['doc_id']
                if doc_id not in seen_indices:
                    expanded_docs.append(self.documents[doc_id])
                    seen_indices.add(doc_id)
        
        # 重新排序所有文档
        all_docs = initial_docs + expanded_docs
        reranked_docs = self._rerank_with_cross_encoder(query, all_docs)
        
        return reranked_docs[:5]  # 返回top5
    
    def qdrant_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        results = self.qdrant.query_text(query, n_results=top_k)
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=doc, metadata=meta))
        return docs

    def _dense_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        return self.qdrant.search_text(query, top_k=top_k)
    
    def _rerank_with_cross_encoder(self, query: str, docs: List[Document]) -> List[Document]:
        """使用交叉编码器重排序"""
        try:
            from sentence_transformers import CrossEncoder
            
            # 使用中文交叉编码器
            reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512,device='cuda:2')
            
            # 准备query-doc对
            pairs = [(query, doc.page_content[:500]) for doc in docs]  # 截断长文本
            
            # 计算重排序分数
            scores = reranker.predict(pairs)
            
            # 按分数排序
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            print(f"⚠️ 重排序失败，使用原顺序: {e}")
            return docs


    def self_rag_query(self, query: str) -> Dict[str, Any]:
        """Self-RAG: 模型自主决定是否需要检索和生成质量"""
        
        # Step 1: 模型自评是否需要检索
        need_retrieval = self._assess_retrieval_need(query)
        
        if not need_retrieval:
            # 直接生成答案
            direct_answer = self._generate_direct_answer(query)
            return {
                'query': query,
                'answer': direct_answer,
                'method': 'direct_generation',
                'retrieval_used': False,
                'confidence': 'medium'
            }
        
        # Step 2: 执行多跳检索
        retrieved_docs = self.multi_hop_retrieval(query)
        
        # Step 3: 生成答案
        rag_answer = self._generate_rag_answer(query, retrieved_docs)
        
        return {
            'query': query,
            'answer': rag_answer,
            'method': 'self_rag',
            'retrieval_used': True,
            'retrieved_docs': retrieved_docs,
        }
    

    def search(self, query: str, k: int = 5) -> List[Document]:
        """分层检索：根据查询返回包含元数据的文档列表"""
        docs: List[Document] = []

        try:
            # 使用分层索引进行检索
            # 根据查询类型自动选择检索策略
            if any(word in query for word in ['诊断', '症状', '表现', '标准']):
                # 诊断类查询，使用级联检索
                docs = self.hierarchical_index.hierarchical_search(query, top_k=k, strategy='cascade')
            elif any(word in query for word in ['治疗', '方法', '药物', '剂量']):
                # 治疗类查询，优先检索句子级别
                docs = self.hierarchical_index.hierarchical_search(query, top_k=k, strategy='hybrid')
            else:
                # 一般查询，使用并行检索
                docs = self.hierarchical_index.hierarchical_search(query, top_k=k, strategy='parallel')
                
        except Exception as e:
            print(f"⚠️ 分层检索失败，回退到基础检索: {e}")
            # 回退到基础 Qdrant 检索
            try:
                docs = self.qdrant_retrieval(query, top_k=k)
            except Exception:
                docs = []

        # 限制数量
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
