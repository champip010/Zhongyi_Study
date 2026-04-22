# src/chroma_manager.py
"""
Chroma向量数据库管理器：负责embedding的存储、检索和元数据管理
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np

class ChromaManager:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "medical_docs", reset: bool = False):
        # 初始化Chroma客户端和集合
        self.client = chromadb.PersistentClient(path=persist_directory)
        if reset:
            self.client.reset() # 重置数据库

        self.collection = self.client.get_or_create_collection(collection_name)

    def add_embeddings_in_batches(self, ids, embeddings, metadatas, documents, batch_size=3000):
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end]
            )

    def query(self, query_embeddings: List[Any], n_results: int = 5, include: List[str] = ["documents", "metadatas", "embeddings"]) -> dict:
        """
        检索最相似的文档，返回原始 Chroma 结果字典，key 为字符串，兼容上游代码。
        """
        try:
            # 统一格式为 list[list[float]]
            processed_embeddings = []
            for emb in query_embeddings:
                if isinstance(emb, np.ndarray):
                    processed_embeddings.append(emb.tolist())
                elif 'torch' in str(type(emb)):
                    processed_embeddings.append(emb.detach().cpu().tolist())
                elif isinstance(emb, list):
                    processed_embeddings.append([float(x) for x in emb])
                else:
                    processed_embeddings.append(list(emb))
            results = self.collection.query(
                query_embeddings=processed_embeddings,
                n_results=n_results,
                include=include
            )
        except Exception as e:
            print(f"❌ ChromaManager.query embedding 格式化失败: {e}")
            raise
        # 直接返回原始结果字典，key 为字符串
        return results
    
    def count(self) -> int:
        return self.collection.count()


    def update_metadata(self, ids: List[str], metadatas: List[Dict[str, Any]]):

        self.collection.update(
            ids=ids,
            metadatas=metadatas,
        )

