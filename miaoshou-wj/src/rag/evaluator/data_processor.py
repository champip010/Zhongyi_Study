# src/data_processor.py (只修改文件读取部分，保留其他所有功能)
import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
import jieba

class SimpleMedicalProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_documents(self, data_dir: str) -> List[Document]:
        """递归加载所有MD文件 - 新增递归功能，保留原有处理逻辑"""
        documents = []
        md_files = []
        
        # 递归查找所有MD文件
        print(f"🔍 递归扫描目录: {data_dir}")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    md_files.append(file_path)
        
        print(f"📚 找到 {len(md_files)} 个MD文件")
        
        # 按文件夹分类统计
        folder_stats = {}
        for file_path in md_files:
            rel_path = os.path.relpath(file_path, data_dir)
            folder = os.path.dirname(rel_path) if os.path.dirname(rel_path) else data_dir
            folder_stats[folder] = folder_stats.get(folder, 0) + 1
        
        print("📁 文件分布:")
        for folder, count in sorted(folder_stats.items()):
            print(f"   {folder}: {count} 个文件")
        
        # 加载所有文件
        loaded_count = 0
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 跳过空文件或过短文件
                if len(content.strip()) < 50:
                    continue
                
                # 提取相对路径信息
                rel_path = os.path.relpath(file_path, data_dir)
                folder_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) else data_dir
                filename = os.path.basename(file_path)
                
                # 提取有用的元数据（保留原逻辑 + 新增文件夹信息）
                metadata = self._extract_metadata(content, filename, folder_name, rel_path)
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                loaded_count += 1
                
                if loaded_count % 100 == 0:
                    print(f"   已加载 {loaded_count} 个文档...")
                
            except Exception as e:
                print(f"⚠️ 加载文件失败 {file_path}: {e}")
                continue
                
        print(f"✅ 成功加载 {len(documents)} 个医学文档")
        return documents
    
    def _extract_metadata(self, content: str, filename: str, folder_name: str = None, rel_path: str = None) -> Dict:
        """从内容中提取有用的元数据 - 保留原逻辑 + 新增文件夹信息"""
        metadata = {'source': filename}
        
        # 新增：文件路径信息
        if folder_name:
            metadata['folder'] = folder_name
        if rel_path:
            metadata['relative_path'] = rel_path
        
        # 保留原有的提取逻辑
        if '诊断' in content or '诊疗' in content:
            metadata['has_diagnosis'] = True
        if '治疗' in content or '干预' in content:
            metadata['has_treatment'] = True  
        if '评价' in content or '评估' in content:
            metadata['has_evaluation'] = True
        if '标准' in content:
            metadata['has_standards'] = True
            
        return metadata
    
    # 保留所有原有方法不变
    def smart_chunk_documents(self, documents: List[Document]) -> List[Document]:
        """智能分块 - 专门针对医学文档优化"""
        
        # 医学文档的层次分隔符
        headers_to_split_on = [
            ("# ", "Header 1"),
            ("## ", "Header 2"), 
            ("### ", "Header 3"),
            ("#### ", "Header 4"),
        ]
        
        # 医学文档常见的分割点
        medical_separators = [
            "\n\n",  # 段落
            "\n",    # 行
            "。",     # 句号
            "；",     # 分号  
            "，",     # 逗号
            " ",      # 空格
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=medical_separators,
            keep_separator=True
        )
        
        all_chunks = []
        
        for doc in documents:
            print(f"处理文档: {doc.metadata.get('relative_path', doc.metadata.get('source', 'unknown'))}")
            
            try:
                # 先按标题分割
                header_splits = markdown_splitter.split_text(doc.page_content)
                
                for split in header_splits:
                    print(split.metadata)
                    # 如果分割后还是太大，继续分割
                    if len(split.page_content) > self.chunk_size:
                        sub_chunks = text_splitter.split_documents([split])
                        
                        for chunk in sub_chunks:
                            chunk.metadata.update(doc.metadata)
                            # 添加段落位置信息
                            chunk.metadata['chunk_type'] = 'sub_chunk'
                            all_chunks.append(chunk)
                    else:
                        split.metadata.update(doc.metadata) 
                        split.metadata['chunk_type'] = 'section'
                        all_chunks.append(split)
                        
            except Exception as e:
                print(f"处理 {doc.metadata.get('source', 'unknown')} 时出错: {e}")
                # 降级处理：直接文本分割
                chunks = text_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata['chunk_type'] = 'fallback'
                    all_chunks.extend(chunks)
        
        print(f"总共生成 {len(all_chunks)} 个chunks")
        return all_chunks
    
    def preview_chunks(self, chunks: List[Document], num_preview: int = 3):
        """预览chunks效果"""
        print(f"\n=== 预览前 {num_preview} 个chunks ===")
        
        for i, chunk in enumerate(chunks[:num_preview]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"来源: {chunk.metadata.get('relative_path', chunk.metadata.get('source', 'unknown'))}")
            print(f"文件夹: {chunk.metadata.get('folder', 'unknown')}")
            print(f"长度: {len(chunk.page_content)} 字符")
            print(f"类型: {chunk.metadata.get('chunk_type', 'unknown')}")
            print(f"内容预览: {chunk.page_content[:200]}...")
            print("-" * 50)