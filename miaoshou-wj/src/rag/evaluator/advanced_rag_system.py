# src/advanced_rag_system.py
import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from vllm import SamplingParams
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json
from chroma_manager import ChromaManager
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from sentence_splitter import SentenceSplitter

# test
class AdvancedMedicalRAG:
    def get_llm_info(self) -> dict:
        """返回当前使用的大模型相关信息"""
        info = {}
        if hasattr(self.llm, 'model_name_or_path'):
            info['model_name'] = getattr(self.llm, 'model_name_or_path', str(self.llm))
        elif hasattr(self.llm, 'model'):
            info['model_name'] = getattr(self.llm, 'model', str(self.llm))
        else:
            info['model_name'] = str(self.llm)
        # 可根据实际llm对象结构补充更多参数
        return info

    def get_embedding_info(self) -> dict:
        """返回当前使用的embedding模型相关信息"""
        info = {}
        if hasattr(self.embedding_model, 'model_name_or_path'):
            info['embedding_model_name'] = getattr(self.embedding_model, 'model_name_or_path', str(self.embedding_model))
        elif hasattr(self.embedding_model, '__str__'):
            info['embedding_model_name'] = str(self.embedding_model)
        else:
            info['embedding_model_name'] = 'unknown'
        # 可根据实际embedding对象结构补充更多参数
        return info
    def __init__(self, llm_model, config: Dict[str, Any]):
        self.llm = llm_model
        self.config = config

        # 初始化Chroma向量数据库
        self.chroma = ChromaManager(persist_directory="./chroma_db/Qwen3-Embedding-0.6B", collection_name="medical_docs",reset = self.config["reset_chroma"])

        # 使用最新的中文医学embedding模型
        qw = "Qwen/Qwen3-Embedding-0.6B"
        bge = "BAAI/bge-large-zh-v1.5"
        self.embedding_model = SentenceTransformer(qw,device = "cuda:3")
        
        # HyDE和Self-RAG的采样参数
        self.hyde_params = SamplingParams(temperature=0.7, max_tokens=300, stop=["<|im_end|>"])
        self.generation_params = SamplingParams(temperature=0.3, max_tokens=800, stop=["<|im_end|>"])
        self.critique_params = SamplingParams(temperature=0.1, max_tokens=100, stop=["<|im_end|>"])
        
        self.documents = []
        self.embeddings = None

        if self.chroma.count() > 0:
            print("Chroma数据库已存在，直接加载文档元数据，无需重建embedding。")
            self._load_documents_from_chroma()
            if self.config["update_chroma_knowledge_connection"]:
                print("更新Chroma知识库连接...")
                self._build_knowledge_connections()
        else:
            print("Chroma数据库为空，需要重建embedding和入库。")
    
    def _load_documents_from_chroma(self):
        """从Chroma加载文档"""
        results = self.chroma.collection.get(include=["documents", "embeddings", "metadatas"])
        self.embeddings = results["embeddings"]
        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        print(f"已从Chroma加载{len(self.documents)}条文档。")

    def build_advanced_index(self, documents: List[Document]):
        print("🔬 构建先进语义索引...")

        self.documents = documents

        # 语义分块 + 上下文增强
        enhanced_chunks = self._semantic_chunking_with_context(documents)
        texts = [doc.page_content for doc in enhanced_chunks]

        print("🧬 生成高质量embeddings并写入Chroma...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        self.embeddings = embeddings
        self.documents = enhanced_chunks

        print("🧬 写入Chroma数据库...")
        # 先写入初始版本（connections为空）

        self.chroma.add_embeddings_in_batches(
            ids=[str(i) for i in range(len(self.documents))],
            embeddings=self.embeddings,
            metadatas=[doc.metadata for doc in self.documents],
            documents=texts,
        )

        # 如果文档较少，构建知识连接
        if len(self.documents) < 20000:
            print("🕸️ 构建知识连接")
            self._build_knowledge_connections()

        else:
            print(f"🕸️ 跳过知识连接构建（文档数量太大: {len(self.documents)}）")
            for doc in self.documents:
                doc.metadata['connections'] = []

        print(f"✅ 索引构建完成: {len(self.documents)} 个增强文档块")


    def _semantic_chunking_with_context(self, documents: List[Document]) -> List[Document]:
        """语义分块 + 上下文增强（批量 embedding 优化版）"""

        enhanced_chunks = []

        for doc in documents:
            # 按段落分割
            splitter = SentenceSplitter(language='zh')
            paragraphs = [
                p for p in splitter.split(doc.page_content) 
                if len(p.strip()) >= 20
            ]
            # paragraphs = [p for p in doc.page_content.split('\n\n') if len(p.strip()) >= 20]
            if not paragraphs:
                continue

            # 一次性批量计算 embedding（减少模型调用次数）
            para_embeddings = self.embedding_model.encode(
                paragraphs, batch_size=16, convert_to_tensor=False
            )

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
        """构建知识连接（使用 Chroma 近邻搜索替代 O(n²)）"""

        for i, doc in enumerate(tqdm(self.documents, desc="构建文档连接")):
            try:
                # 用当前 chunk 内容作为查询，在 Chroma 中找近邻
                query_emb = [self.embeddings[i]]
                results = self.chroma.query(
                    query_embeddings=query_emb,
                    n_results=top_k + 1  # +1 因为会包含自己
                )
                # print(f"构建文档 {i} 的连接...")

                connections = []
                for idx, meta, emb in zip(results["ids"][0], results["metadatas"][0], results["embeddings"][0]):
                    if str(idx) != str(i):  # 排除自己
                        # 用 sentence_transformers 的 cos_sim 计算相似度
                        emb1 = torch.tensor(self.embeddings[i], dtype=torch.float32)
                        emb2 = torch.tensor(emb, dtype=torch.float32)
                        similarity = float(cos_sim(emb1, emb2).item())
                        connections.append({
                            'doc_id': int(idx),
                            'similarity': similarity,
                            'content_type': meta.get('content_type', 'general')
                        })

                # 按similarity降序排序，选top 3
                connections = sorted(connections, key=lambda x: x['similarity'], reverse=True)[:3]
                doc.metadata['connections'] = json.dumps(connections, ensure_ascii=False)
                
                # 重新更新 metadata
                self.chroma.update_metadata(
                    ids=str(i),
                    metadatas=doc.metadata,
                )

            except Exception as e:
                print(f"⚠️ 构建文档 {i} 的连接失败: {e}")
                doc.metadata['connections'] = []

    
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
    
    def chroma_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        query_emb = self.embedding_model.encode([query], convert_to_tensor=False)
        results = self.chroma.query(query_embeddings=query_emb, n_results=top_k)
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
             docs.append(Document(page_content=doc, metadata=meta))
        return docs

    def _dense_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        try:
            return self.chroma_retrieval(query, top_k=top_k)
        
        except Exception as e:
            print(f"Chroma检索失败，回退本地dense检索: {e}")

            if self.embeddings is None:
                return []
            
            # 使用本地dense检索
            query_embedding = self.embedding_model.encode([query])

            # 计算相似度并获取top_k文档
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            return [self.documents[i] for i in top_indices]
    
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
        
        # Step 4: 自评答案质量
        quality_score = self._assess_answer_quality(query, rag_answer, retrieved_docs)
        
        return {
            'query': query,
            'answer': rag_answer,
            'method': 'self_rag',
            'retrieval_used': True,
            'retrieved_docs': retrieved_docs,
            'quality_score': quality_score,
            'confidence': 'high' if quality_score > 0.7 else 'medium'
        }
    
    def _assess_answer_quality(self, query: str, answer: str, docs: List[Document]) -> float:
        """评估答案质量"""
        
        context_content = " ".join([doc.page_content for doc in docs])
        
        quality_prompt = f"""评估以下医学问答的质量（0-1分）：

问题：{query}
答案：{answer}
参考资料长度：{len(context_content)} 字符

评估标准：
- 答案是否准确回答了问题
- 答案是否基于参考资料
- 答案是否专业完整

请只回答一个0-1之间的数字："""
        
        try:
            outputs = self.llm.generate([quality_prompt], self.critique_params)
            score_text = outputs[0].outputs[0].text.strip()
            
            # 提取数字
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            if numbers:
                return float(numbers[0])
            else:
                return 0.5  # 默认中等质量
                
        except Exception as e:
            print(f"⚠️ 质量评估失败: {e}")
            return 0.5