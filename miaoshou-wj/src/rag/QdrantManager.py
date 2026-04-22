
# 设置环境变量确保离线模式
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from socket import timeout
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.models import PointStruct,models
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

import json
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding 

class QdrantManager:
    def __init__(self, collection_name: str = "medical_knowledge",
                 host: str = "127.0.0.1", port: int = 6333,
                 vector_size: int = 1024,
                 embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
                 embedding_device: str = None,
                 reset_qdrant: bool = False):  # BGE-large-zh维度是1024

        self.client = QdrantClient(
            host=host,
            port=port,
            check_compatibility=False,
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        print(f"加载embedding_model_name: {embedding_model_name}")
        self.embedding_device = self.auto_detect_gpu() if embedding_device is None else embedding_device
        print(f"embedding_device: {self.embedding_device}")
        # 使用本地模型加载函数
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.embedding_device)
        print(f"✅ embedding模型加载完成")
        self.sparse_embedding_model = None
        self._ensure_collection()

    def _ensure_collection(self):

        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            if collection_info.config.params.vectors.size != self.vector_size:
                raise ValueError(f"Existing collection has different vector size: {collection_info.config.params.vectors.size}")
        except (UnexpectedResponse, ValueError):
            print(f"Collection {self.collection_name} does not exist")


    def encode_texts(self, texts: List[str], batch_size: int = 16, convert_to_tensor: bool = False):
        if self.embedding_model is None:
            raise ValueError("Embedding 模型未初始化")
        return self.embedding_model.encode(texts, batch_size=batch_size, convert_to_tensor=convert_to_tensor, show_progress_bar=False)

    def add_documents(self, docs: List[Document], embeddings: Optional[List[List[float]]] = None):
        """批量插入文档，若未提供 embeddings 则内部编码；采用小批量 upsert 防止超时"""
        if embeddings is None:
            texts = [doc.page_content for doc in docs]
            embeddings = self.encode_texts(texts, batch_size=16, convert_to_tensor=False)

        # 计算起始 ID，避免与已存在数据冲突
        try:
            start_id = self.count()
        except Exception:
            start_id = 0

        batch_size = 500  # 小批量写入，减小单次请求体积
        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            points: list[PointStruct] = []
            for local_idx, (doc, emb) in enumerate(zip(docs[start:end], embeddings[start:end])):
                global_id = start_id + start + local_idx
                # 向量确保为 Python list[float]
                if isinstance(emb, np.ndarray):
                    vector_list = emb.astype(float).tolist()
                else:
                    vector_list = [float(x) for x in emb]
                # 载荷确保 JSON 可序列化
                metadata = doc.metadata 
                safe_metadata = self._make_json_safe(metadata)
                payload = {
                    "text": doc.page_content,
                    "metadata": safe_metadata,
                    "document_id": f"doc_{global_id}",
                }
                points.append(PointStruct(id=global_id, vector=vector_list, payload=payload))

            # 小批量 upsert（全局客户端已设置较长超时）
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def _make_json_safe(self, obj: Any) -> Any:
        """将 payload/metadata 中的 numpy 类型转换为原生 Python 类型，确保可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        # numpy 类型处理
        try:
            import numpy as _np
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.ndarray,)):
                return obj.tolist()
        except Exception:
            pass
        return obj

    def search_by_vector(self, query_vector: List[float], top_k: int = 5,
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """带过滤条件的语义搜索（输入向量）"""
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                ) for key, value in filter_dict.items()
            ] if filter_dict else None
        )
        
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            Document(
                page_content=hit.payload["text"],
                metadata=hit.payload["metadata"]
            ) for hit in search_results.points
        ]

    def search_text(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """文本查询：内部编码后检索"""
        query_embedding = self.encode_texts([query], convert_to_tensor=False)[0]
        return self.search_by_vector(query_embedding, top_k=top_k, filter_dict=filter_dict)

    def get_document(self, doc_id: int) -> Optional[Document]:
        """按ID获取文档"""
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[doc_id],
            with_payload=True
        )
        if not result:
            return None
        payload = result[0].payload
        return Document(
            page_content=payload["text"],
            metadata=payload.get("metadata", {})
        )

    def update_metadata(self, doc_id: int, metadata: Dict):
        """更新文档元数据"""
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"metadata": metadata},
            points=[doc_id]
        )

    def reset_collection(self):
        """重置当前集合：删除并按当前配置重建"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.client.delete_collection(collection_name="hybrid-search")
        except Exception:
            pass
        # 重建集合
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
                on_disk=True,
            ),
        )

    def count(self) -> int:
        """获取文档数量"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def similarity_search_with_score(
        self, query: str, top_k: int = 5
    ) -> List[tuple[Document, float]]:
        """返回带相似度分数的文档"""
        query_embedding = self.encode_texts([query], convert_to_tensor=False)[0]
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        return [
            (
                Document(
                    page_content=hit.payload["text"],
                    metadata=hit.payload["metadata"]
                ),
                hit.score
            )
            for hit in results.points
        ]

    def get_all(self, batch_size: int = 1000) -> Dict[str, List]:
        """拉取集合内所有文档、向量与元数据，返回 {documents, embeddings, metadatas}"""
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True
            )
            for p in points:
                documents.append(p.payload.get("text", ""))
                metadatas.append(p.payload.get("metadata", {}))
                embeddings.append(p.vector)
            if next_offset is None:
                break

        return {"documents": documents, "embeddings": embeddings, "metadatas": metadatas}

    def get_connected_documents(self, doc_id: int, top_k: int = 3) -> List[Document]:
        """获取知识连接的文档（需预先构建）"""
        doc = self.get_document(doc_id)
        if not doc or "connections" not in doc.metadata:
            return []
        
        connections = doc.metadata["connections"]
        if isinstance(connections, str):
            connections = json.loads(connections)
        
        connected_docs = []
        for conn in connections[:top_k]:
            connected_doc = self.get_document(conn["doc_id"])
            if connected_doc:
                connected_docs.append(connected_doc)
        
        return connected_docs

    def batch_update_connections(self, doc_ids: List[int], is_new_data: bool = False):
        """
        批量更新知识连接（高效实现）
        
        Args:
            doc_ids: 需要更新连接的文档ID列表
            is_new_data: 是否为新数据，如果是则同时更新旧chunk的连接
        """
        batch_size = 100  # Qdrant推荐批量操作
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            
            for doc_id in batch_ids:
                try:
                    # 获取当前文档
                    doc = self.get_document(doc_id)
                    if not doc:
                        continue
                    
                    # 获取相似文档（排除自己）
                    retrieved = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id],
                        with_payload=False,
                        with_vectors=True
                    )
                    if not retrieved:
                        continue
                    doc_vector = retrieved[0].vector

                    # 搜索相似文档
                    similar = self.client.query_points(
                        collection_name=self.collection_name,
                        query=doc_vector,
                        limit=8,  # 增加到10个候选（8个连接+自己+1个缓冲）
                        with_payload=False
                    )
                    
                    # 构建连接信息
                    connections = []
                    for hit in similar.points:
                        if hit.id != doc_id:  # 排除自己
                            # 更严格的相似度过滤：避免创建过于相似的连接
                            if hit.score > 0.7 and hit.score < 0.85:  # 添加上限0.85
                                connections.append({
                                    "doc_id": hit.id,
                                    "similarity": hit.score,
                                    "content_type": self._get_content_type(hit.id)
                                })
                    
                    # 按相似度排序并保留top6连接
                    connections.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    connections = connections[:6]
                    
                    # 更新metadata中的connections字段，不覆盖其他字段
                    current_metadata = doc.metadata.copy()
                    current_metadata['connections'] = connections
                    
                    # 使用set_payload更新metadata
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"metadata": current_metadata},
                        points=[doc_id]
                    )
                    
                except Exception as e:
                    print(f"⚠️ 更新文档 {doc_id} 的连接失败: {e}")
                    continue
            
            print(f"🕸️ 已处理 {min(i + batch_size, len(doc_ids))}/{len(doc_ids)} 个文档的连接...")
        
        # 如果是新数据，更新旧chunk的连接
        if is_new_data:
            self._update_old_connections_simple(doc_ids)
    
    def _update_old_connections_simple(self, new_data_ids: List[int]):
        """简化的旧chunk连接更新：直接比较metadata中的相似度"""
        print("🔄 更新旧chunk的连接...")
        
        # 收集需要更新的旧chunk ID
        old_chunk_updates = {}  # {old_id: [(new_id, similarity), ...]}
        
        # 扫描新数据的连接，找出涉及到的旧chunk
        for new_id in new_data_ids:
            try:
                doc = self.get_document(new_id)
                if doc and doc.metadata:
                    connections = doc.metadata.get("connections", [])
                    for conn in connections:
                        old_id = conn.get("doc_id")
                        if old_id is not None and old_id < new_data_ids[0]:  # 确保是旧chunk
                            if old_id not in old_chunk_updates:
                                old_chunk_updates[old_id] = []
                            old_chunk_updates[old_id].append((new_id, conn.get("similarity", 0)))
            except:
                continue
        
        if not old_chunk_updates:
            print("📊 没有旧chunk需要更新连接")
            return
        
        print(f"📊 需要更新 {len(old_chunk_updates)} 个旧chunk的连接")
        
        # 更新旧chunk的连接
        for old_id, new_connections in old_chunk_updates.items():
            try:
                old_doc = self.get_document(old_id)
                if not old_doc:
                    continue
                
                current_connections = old_doc.metadata.get("connections", [])
                if isinstance(current_connections, str):
                    try:
                        current_connections = json.loads(current_connections)
                    except:
                        current_connections = []
                
                # 检查是否需要更新连接
                updated = False
                for new_id, new_similarity in new_connections:
                    # 查找是否已有这个连接
                    existing_idx = None
                    for i, conn in enumerate(current_connections):
                        if conn.get("doc_id") == new_id:
                            existing_idx = i
                            break
                    
                    # 如果没有这个连接，或者新连接相似度更高，则更新
                    if existing_idx is None or new_similarity > current_connections[existing_idx].get("similarity", 0):
                        if existing_idx is not None:
                            current_connections.pop(existing_idx)
                        
                        current_connections.append({
                            "doc_id": new_id,
                            "similarity": new_similarity,
                            "content_type": self._get_content_type(new_id)
                        })
                        updated = True
                
                if updated:
                    # 重新排序并只保留top6连接（与新建连接保持一致）
                    current_connections.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    current_connections = current_connections[:6]
                    
                    # 更新metadata
                    current_metadata = old_doc.metadata.copy()
                    current_metadata['connections'] = current_connections
                    
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"metadata": current_metadata},
                        points=[old_id]
                    )
                    
            except Exception as e:
                print(f"⚠️ 更新旧chunk {old_id} 的连接失败: {e}")
                continue
        
        print(f"✅ 旧chunk连接更新完成")

    def _get_content_type(self, content: Any) -> str:
        """简单内容类型分类（与 RAG 保持一致逻辑）"""
        # 如果传入的是 doc_id，这里无法直接拿到文本，只能返回 general
        if isinstance(content, int):
            return "general"
        content_lower = str(content).lower()
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

    def query_text(self, query: str, n_results: int = 5) -> Dict[str, List[List[Any]]]:
        """文本检索，返回 documents/metadatas 结构与原有调用兼容"""
        docs = self.search_text(query, top_k=n_results)
        return {
            "documents": [[d.page_content for d in docs]],
            "metadatas": [[d.metadata for d in docs]]
        }

    def create_hybrid_search_collection(self):
        """创建支持混合搜索的集合，包含密集向量、稀疏向量和晚期交互向量"""
        print("🔧 创建混合搜索集合...")
        
        # 不同embedding模型

        dense_embedding_model = self.embedding_model
        bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25",local_files_only=True)
        late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0",local_files_only=True)

        # 获取所有文档
        all_data = self.get_all()
        documents = all_data["documents"]
        dense_embeddings = all_data["embeddings"]
        metadatas = all_data["metadatas"]
        
        print(f"📊 文档数量: {len(documents)}")
        
        # 第一步：先创建数据库集合
        print("📚 第一步：创建混合搜索集合...")
        try:
            # 先尝试删除可能存在的集合
            try:
                self.client.delete_collection("hybrid-search")
                print("🗑️ 删除已存在的集合")
            except Exception:
                pass
            
            # 创建新集合（先不插入数据）
            self.client.create_collection(
                "hybrid-search",
                vectors_config={
                    "dense": models.VectorParams(
                        size=len(dense_embeddings[0]) if dense_embeddings else 1024,
                        distance=models.Distance.COSINE,
                    ),
                    "colbertv2.0": models.VectorParams(  # 使用官方文档中的名称
                        size=128,  # 先使用固定维度，后续会更新
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0)  # 禁用 HNSW 用于重排序
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            print("✅ 混合搜索集合创建成功")
            
            # 验证集合是否真的创建成功
            collection_info = self.client.get_collection("hybrid-search")
            print(f"✅ 集合验证成功，向量配置: {collection_info.config.params.vectors}")
            
        except Exception as e:
            print(f"❌ 集合创建失败: {e}")
            raise

        # 第二步：分批生成 embedding 并插入
        print("📝 第二步：开始分批生成 embedding 并插入...")
        batch_size = 10  # 每批处理200个文档
        
        for start_idx in range(0, len(documents), batch_size):
            end_idx = min(start_idx + batch_size, len(documents))
            current_batch_size = end_idx - start_idx
            batch_num = start_idx // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            print(f"📦 处理批次 {batch_num}/{total_batches}，文档范围: {start_idx}-{end_idx}，数量: {current_batch_size}")
            
            # 生成当前批次的向量
            print(f"🔧 生成批次 {batch_num} 的向量...")
            
            # 批量生成稀疏向量（当前批次）
            current_docs = documents[start_idx:end_idx]
            print("📊 生成稀疏向量...")
            current_bm25_embeddings = list(bm25_embedding_model.embed(current_docs))
            print(f"✅ 稀疏向量生成完成，数量: {len(current_bm25_embeddings)}")
            
            # 批量生成晚期交互向量（当前批次）
            print("📊 生成晚期交互向量...")
            current_late_interaction_embeddings = list(late_interaction_model.embed(current_docs))
            print(f"✅ 晚期交互向量生成完成，数量: {len(current_late_interaction_embeddings)}")
            
            # 准备当前批次的点
            points = []
            for local_idx in range(current_batch_size):
                global_idx = start_idx + local_idx
                
                point = PointStruct(
                    id=global_idx,
                    vector={
                        "dense": dense_embeddings[global_idx],
                        "bm25": current_bm25_embeddings[local_idx].as_object(),
                        "colbertv2.0": current_late_interaction_embeddings[local_idx],  # 名称必须一致
                    },
                    payload={
                        "text": documents[global_idx],
                        "metadata": metadatas[global_idx],
                        "document_id": f"hybrid_doc_{global_idx}"
                    }
                )
                points.append(point)
            
            # 立即插入当前批次
            print(f"📤 插入批次 {batch_num}，文档数量: {len(points)}")
            try:
                self.client.upsert(
                    collection_name="hybrid-search",
                    points=points,
                )
                print(f"✅ 批次 {batch_num} 插入成功，插入 {len(points)} 个文档")
                
                # 验证当前批次插入是否成功
                current_count = self.client.get_collection("hybrid-search").points_count
                print(f"📊 当前集合总文档数: {current_count}")
                
            except Exception as e:
                print(f"❌ 批次 {batch_num} 插入失败: {e}")
                print("⚠️ 继续处理下一批次...")
                continue
            
            # 清理当前批次的向量，释放内存
            del current_bm25_embeddings
            del current_late_interaction_embeddings
            del points
            print(f"🧹 批次 {batch_num} 内存清理完成")
        
        # 最终验证
        try:
            final_count = self.client.get_collection("hybrid-search").points_count
            print(f"🎉 混合搜索集合构建完成！总文档数: {final_count}")
        except Exception as e:
            print(f"⚠️ 最终验证失败: {e}")


  

    def advanced_hybrid_search(self, query, top_k: int = 5):
        """
        三阶段方法：
        1. 密集向量预取
        2. 稀疏向量预取  
        3. 重排序
        """
        try:
            # 检查混合搜索集合是否存在
            try:
                collection_info = self.client.get_collection(collection_name="hybrid-search")
                if collection_info.points_count == 0:
                    print("混合搜索集合为空，正在创建...")
                    self.create_hybrid_search_collection()
            except Exception:
                print("混合搜索集合不存在，正在创建...")
                self.create_hybrid_search_collection()

            # 初始化模型
            if self.sparse_embedding_model is None:
                self.sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25", local_files_only=True)
            
            late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0",local_files_only=True)
            
            print(f"🔍 执行高级混合搜索查询: {query[:50]}...")
            
            # 生成查询的向量表示
            dense_vectors = self.embedding_model.encode([query], convert_to_tensor=False)[0]
            sparse_vectors = list(self.sparse_embedding_model.embed([query]))[0]
            late_vectors = list(late_interaction_model.embed([query]))[0]
            
            # 第一阶段：使用 prefetch 进行混合检索
            prefetch = [
                models.Prefetch(
                    query=dense_vectors,
                    using="dense",
                    limit=30,  # 预取更多候选文档用于重排序
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vectors.as_object()),
                    using="bm25",
                    limit=30,
                ),
            ]
            
            # 第二阶段：使用晚期交互向量进行重排序
            search_results = self.client.query_points(
                collection_name="hybrid-search",
                prefetch=prefetch,
                query=late_vectors,  # 使用晚期交互向量进行重排序
                using="colbertv2.0",  # 名称必须与创建集合时一致
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            # 处理搜索结果
            results = [
                Document(
                    page_content=hit.payload["text"],
                    metadata=hit.payload.get("metadata", {})
                ) for hit in search_results.points
            ]
            
            print(f"✅ 高级混合搜索找到 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            print(f"❌ 高级混合搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
   
    def auto_detect_gpu(self):
        """
        自动检测空闲的GPU设备
        返回可用的GPU设备名称
        """
        try:
            import torch
            import subprocess
            import json
            
            # 检查是否有可用的GPU
            if not torch.cuda.is_available():
                print("⚠️ 未检测到CUDA GPU，使用CPU模式")
                return "cpu"
            
            # 获取CUDA_VISIBLE_DEVICES环境变量
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible_devices:
                print(f"🔍 检测到CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
                # 解析可见的GPU ID列表
                visible_gpu_ids = [int(x.strip()) for x in cuda_visible_devices.split(',') if x.strip().isdigit()]
                print(f"🔍 可见GPU ID列表: {visible_gpu_ids}")
            else:
                # 如果没有设置，则所有GPU都可见
                visible_gpu_ids = list(range(torch.cuda.device_count()))
                print(f"🔍 未设置CUDA_VISIBLE_DEVICES，所有GPU都可见: {visible_gpu_ids}")
            
            if not visible_gpu_ids:
                print("⚠️ 没有可见的GPU，使用CPU模式")
                return "cpu"
            
            # 获取GPU数量（实际可见的数量）
            gpu_count = len(visible_gpu_ids)
            print(f"🔍 实际可见GPU数量: {gpu_count}")
            
            # 使用nvidia-smi获取GPU使用情况，但只关注可见的GPU
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    gpu_info = []
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 4:
                                gpu_id = int(parts[0])
                                # 只处理可见的GPU
                                if gpu_id in visible_gpu_ids:
                                    memory_used = int(parts[1])
                                    memory_total = int(parts[2])
                                    gpu_util = int(parts[3])
                                    
                                    memory_usage = memory_used / memory_total
                                    
                                    gpu_info.append({
                                        'id': gpu_id,
                                        'memory_usage': memory_usage,
                                        'gpu_util': gpu_util,
                                        'score': memory_usage + (gpu_util / 100)  # 综合评分
                                    })
                    
                    if gpu_info:
                        # 按综合评分排序，选择最空闲的GPU
                        gpu_info.sort(key=lambda x: x['score'])
                        
                        best_gpu = gpu_info[0]
                        print(f"✅ 选择GPU {best_gpu['id']} (内存使用: {best_gpu['memory_usage']:.1%}, GPU使用: {best_gpu['gpu_util']}%)")
                        
                        # 返回相对于可见GPU的索引
                        relative_index = visible_gpu_ids.index(best_gpu['id'])
                        return f"cuda:{relative_index}"
                    
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                print("⚠️ 无法获取GPU使用情况，使用启发式选择")
            
            # 启发式选择：尝试找到可用的GPU（使用相对索引）
            for relative_idx, actual_gpu_id in enumerate(visible_gpu_ids):
                try:
                    # 尝试分配少量内存来测试GPU是否可用
                    with torch.cuda.device(relative_idx):
                        test_tensor = torch.zeros(1, device=f'cuda:{relative_idx}')
                        del test_tensor
                        torch.cuda.empty_cache()
                        
                        print(f"✅ 选择GPU {actual_gpu_id} (相对索引: {relative_idx}, 通过可用性测试)")
                        return f"cuda:{relative_idx}"
                        
                except Exception as e:
                    print(f"⚠️ GPU {actual_gpu_id} (相对索引: {relative_idx}) 不可用: {e}")
                    continue
            
            # 如果所有GPU都不可用，使用CPU
            print("⚠️ 所有可见GPU都不可用，使用CPU模式")
            return "cpu"
            
        except ImportError:
            print("⚠️ 未安装PyTorch，使用CPU模式")
            return "cpu"
        except Exception as e:
            print(f"⚠️ GPU检测失败: {e}，使用CPU模式")
            return "cpu"