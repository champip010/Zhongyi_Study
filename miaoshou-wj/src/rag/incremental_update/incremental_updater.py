import os
import json
import time
import torch
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rag.QdrantManager import QdrantManager
from rag.data_processor import SimpleMedicalProcessor
from sentence_transformers.util import cos_sim

class IncrementalUpdater:
    """
    增量更新器：向现有RAG数据库添加新数据
    支持：
    1. 新数据预处理和分块
    2. 构建知识连接
    3. 同时更新普通集合和混合搜索集合
    4. 避免重复数据
    """
    
    def __init__(self, 
                 qdrant_manager: QdrantManager,
                 
                 chunk_size: int = 800,
                 chunk_overlap: int = 200):
        
        self.qdrant_manager = qdrant_manager 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 数据处理器
        self.data_processor = SimpleMedicalProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        print("🔧 增量更新器初始化完成")
    
    def add_new_data(self, 
                     new_data_paths: str,
                     file_limit: int = 0,
                     update_hybrid: bool = True,
                     rebuild_connections: bool = True) -> Dict[str, Any]:
        """
        添加新数据到现有数据库
        
        Args:
            new_data_paths: 新数据路径
            file_limit: 限制处理的文件数量（0表示无限制）
            update_hybrid: 是否同时更新混合搜索集合
            rebuild_connections: 是否重建知识连接
            
        Returns:
            更新结果统计
        """
        print("🚀 开始增量更新流程...")
        
        # 1. 加载和预处理新数据
        print("📚 第一步：加载新数据...")
        new_documents = self.data_processor.load_documents(new_data_paths, file_limit=file_limit)
        new_chunks = self.data_processor.smart_chunk_documents(new_documents)
        
        print(f"📊 新数据统计：")
        print(f"   - 原始文档：{len(new_documents)}")
        print(f"   - 分块后：{len(new_chunks)}")
        
        
        # 2. 语义增强
        print("🧬 第三步：语义增强...")
        enhanced_chunks = self._semantic_chunking_with_context(new_chunks)
        
        # 3. 生成embeddings
        print("🔧 第四步：生成embeddings...")
        texts = [doc.page_content for doc in enhanced_chunks]
        embeddings = self.qdrant_manager.encode_texts(texts, batch_size=16, convert_to_tensor=False)
        
        # 4. 获取当前数据库状态
        current_count = self.qdrant_manager.count()
        print(f"📊 当前数据库文档数：{current_count}")
        
        # 5. 添加到主集合
        print("📤 第五步：添加到主集合...")
        self.qdrant_manager.add_documents(
            docs=enhanced_chunks,
            embeddings=embeddings
        )
        
        # 6. 构建知识连接（如果启用）
        if rebuild_connections:
            print("🕸️ 第六步：构建知识连接...")
            self._build_knowledge_connections_for_new_data(enhanced_chunks)
        
        # 7. 更新混合搜索集合（如果启用）
        if update_hybrid:
            print("🔀 第七步：更新混合搜索集合...")
            self._update_hybrid_collection(enhanced_chunks, embeddings)
        
        # 9. 最终验证
        final_count = self.qdrant_manager.count()
        added_count = final_count - current_count
        
        print(f"🎉 增量更新完成！")
        print(f"   - 新增文档：{added_count}")
        print(f"   - 总文档数：{final_count}")
        
        return {
            "status": "success",
            "added_count": added_count,
            "total_count": final_count,
            "enhanced_chunks": len(enhanced_chunks)
        }
    
    
    
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
        """中文分句（复用RAG系统的逻辑）"""
        import jieba
        
        sentence_endings = ['。', '！', '？', '；', '\n\n', '\n']
        words = list(jieba.cut(text))
        
        sentences = []
        current_sentence = ""
        
        for word in words:
            current_sentence += word
            
            if any(ending in current_sentence for ending in sentence_endings):
                for ending in sentence_endings:
                    if ending in current_sentence:
                        parts = current_sentence.split(ending)
                        for i, part in enumerate(parts[:-1]):
                            if part.strip():
                                sentences.append(part.strip() + ending)
                        current_sentence = parts[-1]
                        break
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 智能合并短句子
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            
            if len(current.strip()) < 20 and i + 1 < len(sentences):
                merged = current + " " + sentences[i + 1]
                merged_sentences.append(merged)
                i += 2
            else:
                merged_sentences.append(current)
                i += 1
        
        return merged_sentences
    
    def _add_context(self, chunk_content: str, original_doc: Document, position: int) -> Document:
        """为chunk添加上下文信息（复用RAG系统的逻辑）"""
        
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
            'medical_entities': ",".join(self._extract_medical_entities(chunk_content)),
            'is_incremental': True,  # 标记为增量更新
            'update_timestamp': str(int(time.time()))
        })
        
        return Document(
            page_content=enhanced_content,
            metadata=enhanced_metadata
        )
    
    def _classify_content_type(self, content: str) -> str:
        """分类内容类型（复用RAG系统的逻辑）"""
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
    
    def _extract_medical_entities(self, content: str) -> List[str]:
        """提取医学实体（复用RAG系统的逻辑）"""
        medical_terms = [
            '失眠', '多梦', '早醒', '入睡困难', '睡眠质量',
            'PSQI', '帕金森病', '镇静催眠药', '戒断综合征',
            '肝气郁结', '心脾两虚', '心肾不交', '阴虚火旺',
            '高血压', '糖尿病', '冠心病', '心肌梗死', '脑卒中',
            '肺炎', '哮喘', '肝炎', '肾炎', '胃炎', '肠炎',
            '发热', '疼痛', '咳嗽', '呼吸困难', '头痛', '恶心',
            '手术', '药物治疗', '化疗', '放疗', '康复',
            'CT', 'MRI', 'X光', '超声', '心电图', '血检'
        ]
        
        found_terms = []
        for term in medical_terms:
            if term in content:
                found_terms.append(term)
        
        return found_terms
    
    def _update_hybrid_collection(self, enhanced_chunks: List[Document], embeddings: List[List[float]]):
        """更新混合搜索集合"""
        print("🔀 开始更新混合搜索集合...")
        
        try:
            # 检查混合搜索集合是否存在
            try:
                collection_info = self.qdrant_manager.client.get_collection("hybrid-search")
                print(f"📊 混合搜索集合已存在，当前文档数：{collection_info.points_count}")
            except Exception:
                print("📚 混合搜索集合不存在，正在创建...")
                self.qdrant_manager.create_hybrid_search_collection()
                return
            
            # 获取当前集合的文档数量作为起始ID
            current_count = collection_info.points_count
            print(f"📊 当前混合搜索集合文档数：{current_count}")
            
            # 分批处理新数据
            batch_size = 10
            for start_idx in range(0, len(enhanced_chunks), batch_size):
                end_idx = min(start_idx + batch_size, len(enhanced_chunks))
                current_batch_size = end_idx - start_idx
                batch_num = start_idx // batch_size + 1
                total_batches = (len(enhanced_chunks) + batch_size - 1) // batch_size
                
                print(f"📦 处理混合搜索批次 {batch_num}/{total_batches}，文档范围: {start_idx}-{end_idx}")
                
                # 生成当前批次的向量
                current_docs = [chunk.page_content for chunk in enhanced_chunks[start_idx:end_idx]]
                
                # 生成稀疏向量
                from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
                bm25_model = SparseTextEmbedding("Qdrant/bm25")
                late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
                
                current_bm25_embeddings = list(bm25_model.embed(current_docs))
                current_late_interaction_embeddings = list(late_interaction_model.embed(current_docs))
                
                # 准备点数据
                points = []
                for local_idx in range(current_batch_size):
                    global_idx = current_count + start_idx + local_idx
                    
                    point = {
                        "id": global_idx,
                        "vector": {
                            "dense": embeddings[start_idx + local_idx],
                            "bm25": current_bm25_embeddings[local_idx].as_object(),
                            "colbertv2.0": current_late_interaction_embeddings[local_idx],
                        },
                        "payload": {
                            "text": current_docs[local_idx],
                            "metadata": enhanced_chunks[start_idx + local_idx].metadata,
                            "document_id": f"hybrid_doc_{global_idx}"
                        }
                    }
                    points.append(point)
                
                # 插入到混合搜索集合
                try:
                    self.qdrant_manager.client.upsert(
                        collection_name="hybrid-search",
                        points=points,
                    )
                    print(f"✅ 混合搜索批次 {batch_num} 插入成功")
                except Exception as e:
                    print(f"❌ 混合搜索批次 {batch_num} 插入失败: {e}")
                    continue
                
                # 清理内存
                del current_bm25_embeddings
                del current_late_interaction_embeddings
                del points
            
            # 最终验证
            final_count = self.qdrant_manager.client.get_collection("hybrid-search").points_count
            print(f"🎉 混合搜索集合更新完成！总文档数：{final_count}")
            
        except Exception as e:
            print(f"❌ 更新混合搜索集合失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_knowledge_connections_for_new_data(self, enhanced_chunks: List[Document]):
        """为新数据构建知识连接"""
        print("🕸️ 开始为新数据构建知识连接...")
        
        try:
            # 获取新数据的ID范围
            current_count = self.qdrant_manager.count()
            new_data_start_id = current_count - len(enhanced_chunks)
            new_data_ids = list(range(new_data_start_id, current_count))
            
            print(f"📊 新数据ID范围：{new_data_start_id} - {current_count-1}")
            
            # 使用QdrantManager的批量更新连接方法，标记为新数据
            self.qdrant_manager.batch_update_connections(new_data_ids, is_new_data=True)
            
            print("✅ 新数据知识连接构建完成")
            
        except Exception as e:
            print(f"❌ 构建知识连接失败: {e}")
            import traceback
            traceback.print_exc()
    
    def get_update_status(self) -> Dict[str, Any]:
        """获取更新状态信息"""
        try:
            main_count = self.qdrant_manager.count()
            
            # 检查混合搜索集合
            try:
                hybrid_info = self.qdrant_manager.client.get_collection("hybrid-search")
                hybrid_count = hybrid_info.points_count
                hybrid_exists = True
            except Exception:
                hybrid_count = 0
                hybrid_exists = False
            
            return {
                "main_collection": {
                    "exists": True,
                    "document_count": main_count
                },
                "hybrid_collection": {
                    "exists": hybrid_exists,
                    "document_count": hybrid_count
                },
                "last_update": "N/A"  # 可以扩展为记录最后更新时间
            }
        except Exception as e:
            return {
                "error": str(e),
                "main_collection": {"exists": False, "document_count": 0},
                "hybrid_collection": {"exists": False, "document_count": 0}
            }
    
    def cleanup_old_data(self, days_threshold: int = 30):
        """清理旧数据（可选功能）"""
        print(f"🧹 清理 {days_threshold} 天前的旧数据...")
        
        # 这里可以实现基于时间戳的数据清理逻辑
        # 需要metadata中包含时间信息
        print("⚠️ 数据清理功能待实现")
        pass


