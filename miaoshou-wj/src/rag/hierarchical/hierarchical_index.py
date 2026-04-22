"""
分层索引管理器
实现文档 -> 段落 -> 句子的三层索引结构
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from sentence_transformers.util import cos_sim
import torch
import jieba
from tqdm import tqdm


class HierarchicalIndexManager:
    def __init__(self, qdrant_manager):
        self.qdrant_manager = qdrant_manager
        self.levels = {
            'document': 'medical_documents',      # 文档级别
            'paragraph': 'medical_paragraphs',    # 段落级别  
            'sentence': 'medical_sentences'       # 句子级别
        }
        
    def build_hierarchical_index(self, documents: List[Document]) -> Dict[str, Any]:
        """构建分层索引"""
        print("🏗️ 开始构建分层索引...")
        
        # 第一层：文档级别索引
        print("📚 构建文档级别索引...")
        doc_embeddings = self._build_document_level(documents)
        
        # 第二层：段落级别索引
        print("📖 构建段落级别索引...")
        para_embeddings = self._build_paragraph_level(documents)
        
        # 第三层：句子级别索引
        print("🔤 构建句子级别索引...")
        sent_embeddings = self._build_sentence_level(documents)
        
        # 建立层次关系
        print("🔗 建立层次关系...")
        self._build_hierarchical_relations(documents, para_embeddings, sent_embeddings)
        
        print("✅ 分层索引构建完成！")
        return {
            'documents': len(documents),
            'paragraphs': len(para_embeddings),
            'sentences': len(sent_embeddings)
        }
    
    def _build_document_level(self, documents: List[Document]) -> List[List[float]]:
        """构建文档级别索引"""
        # 提取文档标题和摘要
        doc_texts = []
        for doc in documents:
            # 提取前200字符作为文档摘要
            summary = doc.page_content[:200] + "..."
            title = doc.metadata.get('source', '未知文档')
            doc_text = f"标题：{title}\n摘要：{summary}"
            doc_texts.append(doc_text)
        
        # 生成文档级别 embedding
        doc_embeddings = self.qdrant_manager.encode_texts(doc_texts, batch_size=8)
        
        # 存储到 Qdrant
        self._store_to_qdrant('document', documents, doc_embeddings, doc_texts)
        
        return doc_embeddings
    
    def _build_paragraph_level(self, documents: List[Document]) -> List[List[float]]:
        """构建段落级别索引"""
        all_paragraphs = []
        all_para_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            # 按段落分割（基于换行符和标题）
            paragraphs = self._split_into_paragraphs(doc.page_content)
            
            for para_idx, para in enumerate(paragraphs):
                if len(para.strip()) >= 50:  # 过滤太短的段落
                    all_paragraphs.append(para)
                    all_para_metadata.append({
                        'document_id': doc_idx,
                        'paragraph_id': para_idx,
                        'source': doc.metadata.get('source', '未知'),
                        'content_type': self._classify_content_type(para),
                        'length': len(para)
                    })
        
        # 生成段落级别 embedding
        para_embeddings = self.qdrant_manager.encode_texts(all_paragraphs, batch_size=16)
        
        # 存储到 Qdrant
        self._store_to_qdrant('paragraph', all_paragraphs, para_embeddings, all_para_metadata)
        
        return para_embeddings
    
    def _build_sentence_level(self, documents: List[Document]) -> List[List[float]]:
        """构建句子级别索引"""
        all_sentences = []
        all_sent_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            # 按句子分割
            sentences = self._chinese_sentence_split(doc.page_content)
            
            for sent_idx, sent in enumerate(sentences):
                if len(sent.strip()) >= 20:  # 过滤太短的句子
                    all_sentences.append(sent)
                    all_sent_metadata.append({
                        'document_id': doc_idx,
                        'sentence_id': sent_idx,
                        'source': doc.metadata.get('source', '未知'),
                        'content_type': self._classify_content_type(sent),
                        'length': len(sent)
                    })
        
        # 生成句子级别 embedding
        sent_embeddings = self.qdrant_manager.encode_texts(all_sentences, batch_size=32)
        
        # 存储到 Qdrant
        self._store_to_qdrant('sentence', all_sentences, sent_embeddings, all_sent_metadata)
        
        return sent_embeddings
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        # 基于换行符和标题分割
        lines = text.split('\n')
        paragraphs = []
        current_para = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append(current_para)
                    current_para = ""
            elif line.startswith('#'):  # 标题行
                if current_para:
                    paragraphs.append(current_para)
                current_para = line
            else:
                if current_para:
                    current_para += "\n" + line
                else:
                    current_para = line
        
        if current_para:
            paragraphs.append(current_para)
        
        return [p for p in paragraphs if len(p.strip()) >= 50]
    
    def _chinese_sentence_split(self, text: str) -> List[str]:
        """中文分句"""
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
        
        return [s for s in sentences if len(s.strip()) >= 20]
    
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
    
    def _store_to_qdrant(self, level: str, texts: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """存储到 Qdrant"""
        collection_name = self.levels[level]
        
        # 创建集合（如果不存在）
        self._ensure_collection(collection_name, len(embeddings[0]))
        
        # 准备数据点
        points = []
        for idx, (text, emb, meta) in enumerate(zip(texts, embeddings, metadata)):
            payload = {
                "text": text,
                "metadata": meta,
                "level": level,
                "id": f"{level}_{idx}"
            }
            points.append({"id": idx, "vector": emb, "payload": payload})
        
        # 批量插入
        self.qdrant_manager.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
    
    def _ensure_collection(self, collection_name: str, vector_size: int):
        """确保集合存在"""
        try:
            collection_info = self.qdrant_manager.client.get_collection(collection_name)
            if collection_info.vectors_config.params.size != vector_size:
                raise ValueError(f"Existing collection has different vector size")
        except:
            from qdrant_client.http.models import Distance, VectorParams
            self.qdrant_manager.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=Distance.COSINE,
                    on_disk=True
                )
            )
    
    def _build_hierarchical_relations(self, documents: List[Document], para_embeddings: List, sent_embeddings: List):
        """建立层次关系"""
        # 这里可以建立文档-段落-句子的关系映射
        # 存储到元数据中，便于后续检索时使用
        pass
    
    def hierarchical_search(self, query: str, top_k: int = 5, strategy: str = 'cascade') -> List[Document]:
        """分层检索"""
        if strategy == 'cascade':
            return self._cascade_search(query, top_k)
        elif strategy == 'parallel':
            return self._parallel_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)
    
    def _cascade_search(self, query: str, top_k: int) -> List[Document]:
        """级联检索：先查文档，再查段落，最后查句子"""
        # 1. 文档级别检索
        doc_results = self._search_level('document', query, top_k=2)
        
        # 2. 基于文档结果，检索相关段落
        para_results = self._search_level('paragraph', query, top_k=top_k)
        
        # 3. 基于段落结果，检索相关句子
        sent_results = self._search_level('sentence', query, top_k=top_k*2)
        
        # 合并结果并重排序
        all_results = doc_results + para_results + sent_results
        return self._rerank_results(query, all_results, top_k)
    
    def _parallel_search(self, query: str, top_k: int) -> List[Document]:
        """并行检索：同时检索三个层次"""
        doc_results = self._search_level('document', query, top_k=top_k//3)
        para_results = self._search_level('paragraph', query, top_k=top_k//3)
        sent_results = self._search_level('sentence', query, top_k=top_k//3)
        
        all_results = doc_results + para_results + sent_results
        return self._rerank_results(query, all_results, top_k)
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Document]:
        """混合检索：根据查询类型选择策略"""
        # 分析查询类型
        if any(word in query for word in ['诊断', '症状', '表现']):
            # 诊断类查询，优先检索段落级别
            return self._search_level('paragraph', query, top_k)
        elif any(word in query for word in ['治疗', '方法', '药物']):
            # 治疗类查询，优先检索句子级别
            return self._search_level('sentence', query, top_k)
        else:
            # 一般查询，使用级联检索
            return self._cascade_search(query, top_k)
    
    def _search_level(self, level: str, query: str, top_k: int) -> List[Document]:
        """在指定层次检索"""
        collection_name = self.levels[level]
        
        # 编码查询
        query_embedding = self.qdrant_manager.encode_texts([query], convert_to_tensor=False)[0]
        
        # 检索
        results = self.qdrant_manager.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # 转换为 Document 对象
        documents = []
        for hit in results:
            doc = Document(
                page_content=hit.payload["text"],
                metadata=hit.payload["metadata"]
            )
            doc.metadata['level'] = level
            doc.metadata['score'] = hit.score
            documents.append(doc)
        
        return documents
    
    def _rerank_results(self, query: str, results: List[Document], top_k: int) -> List[Document]:
        """重排序结果"""
        # 基于层次和相似度分数重排序
        for doc in results:
            # 层次权重：句子 > 段落 > 文档
            level_weights = {'sentence': 3, 'paragraph': 2, 'document': 1}
            level_weight = level_weights.get(doc.metadata.get('level', 'sentence'), 1)
            
            # 调整分数
            doc.metadata['adjusted_score'] = doc.metadata.get('score', 0) * level_weight
        
        # 按调整后的分数排序
        results.sort(key=lambda x: x.metadata.get('adjusted_score', 0), reverse=True)
        
        return results[:top_k]
