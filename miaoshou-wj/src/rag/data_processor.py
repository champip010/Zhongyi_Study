import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import jieba
from tqdm import tqdm

# PDF处理相关导入
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import UnstructuredPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF处理库未安装，PDF功能将不可用")

class SimpleMedicalProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, data_dir: str, file_limit: int = 0) -> List[Document]:
        """递归加载并清理所有Markdown和PDF文件；可选限制文件数量以加快小样本测试"""
        documents = []
        md_files = []
        pdf_files = []

        # 递归查找所有MD和PDF文件
        print(f"🔍 递归扫描目录: {data_dir}")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    md_files.append(file_path)
                elif file.endswith('.pdf') and PDF_AVAILABLE:
                    file_path = os.path.join(root, file)
                    pdf_files.append(file_path)
                
                if file_limit and len(md_files) + len(pdf_files) >= file_limit:
                    break
            if file_limit and len(md_files) + len(pdf_files) >= file_limit:
                break

        print(f"�� 找到 {len(md_files)} 个MD文件，{len(pdf_files)} 个PDF文件")

        # 加载Markdown文件
        loaded_count = 0
        for file_path in tqdm(md_files, desc="加载Markdown文件", unit="file"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()

                cleaned_content, clean_log = self.clean_medical_md_content(raw_content, os.path.basename(file_path))

                if len(cleaned_content.strip()) < 50:
                    print(f"⚠️ 跳过空文件 (清洗后): {file_path}")
                    continue

                # 元数据
                rel_path = os.path.relpath(file_path, data_dir)
                folder_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) else data_dir
                filename = os.path.basename(file_path)

                metadata = self._extract_metadata(cleaned_content, filename, folder_name, rel_path)
                metadata['file_type'] = 'markdown'  # 标记文件类型

                doc = Document(page_content=cleaned_content, metadata=metadata)
                documents.append(doc)
                loaded_count += 1

                if loaded_count % 50 == 0:
                    print(f"   已加载 {loaded_count} 个文档...")

            except Exception as e:
                print(f"⚠️ 加载Markdown文件失败 {file_path}: {e}")
                continue

        # 加载PDF文件
        if PDF_AVAILABLE:
            for file_path in tqdm(pdf_files, desc="加载PDF文件", unit="file"):
                try:
                    pdf_docs = self._load_and_clean_pdf(file_path, data_dir)
                    if pdf_docs:
                        documents.extend(pdf_docs)
                        loaded_count += len(pdf_docs)
                except Exception as e:
                    print(f"⚠️ 加载PDF文件失败 {file_path}: {e}")
                    continue

        print(f"✅ 成功加载 {len(documents)} 个医学文档")
        return documents

    def _load_and_clean_pdf(self, file_path: str, data_dir: str) -> List[Document]:
        """加载并清理PDF文件"""
        try:
            # 尝试使用PyPDFLoader（更快但功能有限）
            try:
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
            except Exception:
                # 降级到UnstructuredPDFLoader（更稳定但较慢）
                loader = UnstructuredPDFLoader(file_path)
                pdf_docs = loader.load()

            # 清理和合并PDF内容
            cleaned_docs = self._clean_and_merge_pdf_docs(pdf_docs, file_path, data_dir)
            return cleaned_docs

        except Exception as e:
            print(f"⚠️ PDF加载失败 {file_path}: {e}")
            return []

    def _clean_and_merge_pdf_docs(self, pdf_docs: List[Document], file_path: str, data_dir: str) -> List[Document]:
        """清理和合并PDF文档"""
        if not pdf_docs:
            return []

        # 提取元数据
        rel_path = os.path.relpath(file_path, data_dir)
        folder_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) else data_dir
        filename = os.path.basename(file_path)

        # 合并所有页面内容
        all_content = ""
        for doc in pdf_docs:
            content = doc.page_content.strip()
            if content:
                all_content += content + "\n\n"

        # 清理PDF内容
        cleaned_content = self.clean_medical_pdf_content(all_content)
        
        if len(cleaned_content.strip()) < 100:
            return []

        # 创建合并后的文档
        metadata = self._extract_metadata(cleaned_content, filename, folder_name, rel_path)
        metadata['file_type'] = 'pdf'
        metadata['original_pages'] = len(pdf_docs)

        merged_doc = Document(page_content=cleaned_content, metadata=metadata)
        return [merged_doc]

    def clean_medical_pdf_content(self, content: str) -> str:
        """清理PDF内容，去除格式问题"""
        # 去除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        
        # 去除页眉页脚（通常是页码、标准号等）
        content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'ICS \*\*\.\*\*\*\.\*\*', '', content)
        content = re.sub(r'C\*\*', '', content)
        content = re.sub(r'T/CACM \*\*\*\*－20\*\*', '', content)
        
        # 清理换行符，保持段落结构
        content = re.sub(r'(?<=\w)\n(?=\w)', ' ', content)  # 单词间的换行改为空格
        
        # 去除PDF特有的格式字符和引用标记
        content = re.sub(r'\[.*?\]', '', content)
        content = re.sub(r'（文件类型：.*?）', '', content)
        content = re.sub(r'20\*\*-\*\*-\*\*发布.*?实施', '', content)
        
        # 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()

    def _extract_metadata(self, content: str, filename: str, folder_name: str = None, rel_path: str = None) -> Dict:
        """从内容中提取元数据"""
        metadata = {'source': filename}

        # 文件路径信息
        if folder_name:
            metadata['folder'] = folder_name
        if rel_path:
            metadata['relative_path'] = rel_path

        # 简单内容特征提取
        if '诊断' in content or '诊疗' in content:
            metadata['has_diagnosis'] = True
        if '治疗' in content or '干预' in content:
            metadata['has_treatment'] = True
        if '评价' in content or '评估' in content:
            metadata['has_evaluation'] = True
        if '标准' in content:
            metadata['has_standards'] = True

        return metadata

    def clean_medical_md_content(self, content: str, filename: str = "") -> (str, Dict):
        """
        医学Markdown清洗：确保可被 MarkdownHeaderTextSplitter 正确识别
        返回：清洗后的文本 + 清洗日志
        """
        clean_log = {
            'original_length': len(content),
            'header_changes': [],
            'other_changes': []
        }

        # 1. 移除图片占位符 ![...](...)
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        # 2. 标题标准化
        # 2.1 修正数字型标题： "# 1、 定义" → "# 1. 定义"
        def _fix_number_headers(match):
            orig = match.group(0)
            new = f"{match.group(1)} {match.group(2)}. {match.group(3)}"
            clean_log['header_changes'].append((orig, new))
            return new

        content = re.sub(
            r'^(#{1,4})\s*(\d+)[、\.]?\s*([^\n]+)',
            _fix_number_headers,
            content,
            flags=re.MULTILINE
        )

        # 2.2 修正括号序号标题 "(一）特征表现" → "## (一) 特征表现"
        content = re.sub(
            r'^(#{1,4})\s*[（(]([一二三四五六七八九十]+)[)）]\s*([^\n]+)',
            r'\1 (\2) \3',
            content,
            flags=re.MULTILINE
        )

        # 2.3 去掉标题中的特殊符号（如 ※）
        content = re.sub(r'^#{1,4}\s*※+', lambda m: m.group(0).replace('※', ''), content, flags=re.MULTILINE)

        # 3. 列表符号标准化
        # • 或 ①②③ 转换为 "- "
        content = re.sub(r'^\s*[•●]\s*', '- ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*', '- ', content, flags=re.MULTILINE)

        # 4. 合并多余空行（保留最多2个）
        content = re.sub(r'\n{3,}', '\n\n', content)

        clean_log['final_length'] = len(content)
        return content, clean_log

    def smart_chunk_documents(self, documents: List[Document]) -> List[Document]:
        """智能分块 - 针对医学文档优化，使用简单段落分割"""

        medical_separators = [
            "\n\n",  # 段落
            "\n",    # 行
            "。",     # 句号
            "；",     # 分号
            "，",     # 逗号
            " ",      # 空格
        ]

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
                # 先提取参考文献和附件，单独处理
                main_content, special_chunks = self._extract_special_sections(doc)
                
                # 处理主要内容
                if main_content.strip():
                    chunks = text_splitter.split_documents([Document(page_content=main_content, metadata=doc.metadata)])
                    
                    # 为每个chunk添加元数据
                    for chunk in chunks:
                        chunk.metadata.update(doc.metadata)
                        chunk.metadata['chunk_type'] = 'text_split'
                        all_chunks.append(chunk)
                
                # 添加特殊chunk（参考文献、附件等）
                all_chunks.extend(special_chunks)

            except Exception as e:
                print(f"处理 {doc.metadata.get('source', 'unknown')} 时出错: {e}")
                # 降级处理：按段落分割
                paragraphs = doc.page_content.split('\n\n')
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        chunk = Document(
                            page_content=para.strip(),
                            metadata=doc.metadata.copy()
                        )
                        chunk.metadata['chunk_type'] = 'fallback_paragraph'
                        all_chunks.append(chunk)

        print(f"总共生成 {len(all_chunks)} 个chunks")
        print(f"chunks的平均长度: {sum(len(chunk.page_content) for chunk in all_chunks) / len(all_chunks)}")
        return all_chunks

    def _extract_special_sections(self, doc: Document) -> tuple[str, List[Document]]:
        """提取参考文献和附件等特殊部分，返回主要内容和特殊chunk列表"""
        content = doc.page_content
        special_chunks = []
        
        # 查找参考文献部分
        ref_patterns = [
            r'(#\s*【参考文献】.*?)(?=#|$)',
            r'(#\s*参考文献.*?)(?=#|$)',
            r'(#\s*References.*?)(?=#|$)',
        ]
        
        # 查找附件部分
        attachment_patterns = [
            r'(#\s*附件一.*?)(?=#|$)',
            r'(#\s*附件二.*?)(?=#|$)',
            r'(#\s*附件三.*?)(?=#|$)',
            r'(#\s*附件.*?)(?=#|$)',
        ]
        
        # 提取参考文献
        for pattern in ref_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if match.strip():
                    ref_chunk = Document(
                        page_content=match.strip(),
                        metadata=doc.metadata.copy()
                    )
                    ref_chunk.metadata['chunk_type'] = 'references'
                    ref_chunk.metadata['section_type'] = 'references'
                    special_chunks.append(ref_chunk)
                    # 从主内容中移除
                    content = content.replace(match, '')
        
        # 提取附件
        for pattern in attachment_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if match.strip():
                    attachment_chunk = Document(
                        page_content=match.strip(),
                        metadata=doc.metadata.copy()
                    )
                    attachment_chunk.metadata['chunk_type'] = 'attachment'
                    attachment_chunk.metadata['section_type'] = 'attachment'
                    special_chunks.append(attachment_chunk)
                    # 从主内容中移除
                    content = content.replace(match, '')
        
        # 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content).strip()
        
        # print(f"✅ 提取了 {len(special_chunks)} 个特殊部分")
        return content, special_chunks


    def preview_chunks(self, chunks: List[Document], num_preview: int = 3):
        """预览chunks效果"""
        print(f"\n=== 预览前 {num_preview} 个chunks ===")

        for i, chunk in enumerate(chunks[:num_preview]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"来源: {chunk.metadata.get('relative_path', chunk.metadata.get('source', 'unknown'))}")
            print(f"文件夹: {chunk.metadata.get('folder', 'unknown')}")
            print(f"文件类型: {chunk.metadata.get('file_type', 'unknown')}")
            print(f"长度: {len(chunk.page_content)} 字符")
            print(f"类型: {chunk.metadata.get('chunk_type', 'unknown')}")
            print(f"内容预览: {chunk.page_content[:200]}...")
            print("-" * 50)