#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统对话接口
提供基于检索增强生成的智能问答功能
"""

from typing import List
from langchain.schema import Document
from vllm import SamplingParams


class ChatInterface:
    """RAG对话接口类"""
    
    def __init__(self, rag_system):
        """
        初始化对话接口
        
        Args:
            rag_system: RAG系统实例
        """
        self.rag_system = rag_system
        self.llm = rag_system.llm
        
        # 用户偏好设置
        self.use_hybrid_retrieval = True  # 默认使用混合检索
        self.show_documents = False       # 默认不显示文档内容
        
        # 历史记录管理
        self.max_history = 10            # 最大历史记录条数（根据显存调整）
        self.conversation_history = []   # 对话历史记录
    
    def chat_with_rag(self, query: str, k: int = 5, use_hybrid_retrieval: bool = False, show_documents: bool = False) -> str:
        """
        基于RAG的对话接口
        
        Args:
            query: 用户问题
            k: 检索文档数量
            use_hybrid_retrieval: 是否使用混合检索
            show_documents: 是否在回答中显示检索到的文档内容
            
        Returns:
            模型回答
        """
        if not self.llm:
            return "❌ 错误：LLM模型未加载，无法生成回答"
        
        try:
            # 1. 检索相关文档
            print(f"🔍 检索相关文档...")
            docs = self.rag_system.search(query, k=k, use_hybrid_retrieval=use_hybrid_retrieval)
            
            if not docs:
                return "❌ 抱歉，没有找到相关的文档信息"
            
            # 2. 构建上下文
            context = self._build_context_from_docs(docs)
            
            # 3. 构建提示词
            prompt = self._build_chat_prompt(query, context, show_documents)
            
            # 4. 生成回答
            print(f"🤖 生成回答...")
            response = self._generate_response(prompt)
            
            # 5. 如果需要显示文档内容，在回答后添加文档信息
            if show_documents:
                response += f"\n\n📚 检索到的相关文档：\n{self._format_documents_for_display(docs)}"
            
            # 6. 管理历史记录
            self._manage_history(query, response)
            
            return response
            
        except Exception as e:
            return f"❌ 生成回答时发生错误：{e}"

    def _build_context_from_docs(self, docs: List[Document]) -> str:
        """从检索到的文档构建上下文"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            # 提取文档信息
            content = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            folder = doc.metadata.get('folder', '')
            
            # 构建文档摘要
            doc_info = f"【文档{i}】来源：{source}"
            if folder:
                doc_info += f" | 分类：{folder}"
            
            # 截断过长的内容
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_parts.append(f"{doc_info}\n{content}\n")
        
        return "\n".join(context_parts)

    def _build_chat_prompt(self, query: str, context: str, show_documents: bool) -> str:
        """构建对话提示词"""
        # 获取历史上下文
        history_context = self._get_context_from_history()
        
        prompt = f"""你是一个专业的医学知识助手，基于以下检索到的医学文档来回答用户问题。

{history_context}

用户问题：{query}

相关文档信息：
{context}

请基于上述文档信息和对话历史，用专业但易懂的语言回答用户问题。要求：
1. 回答要准确、专业
2. 语言要通俗易懂
3. 如果文档信息不足，请说明并建议用户提供更多信息
4. 回答要结构清晰，重点突出
5. 保持对话的连贯性
6. 请提供完整的回答，不要中途停止

回答："""
        
        return prompt

    def _generate_response(self, prompt: str) -> str:
        """使用LLM生成回答"""
        try:
            # 设置生成参数
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=10000,
                stop=["<|im_end|>", "用户：", "助手："]  # 移除 "\n\n" 避免过早停止
            )
            
            # 生成回答
            outputs = self.llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            return response
            
        except Exception as e:
            return f"❌ 模型生成失败：{e}"

    def start_chat_service(self):
        """启动交互式对话服务"""
        if not self.llm:
            print("❌ 错误：LLM模型未加载，无法启动对话服务")
            return
        
        print("🚀 启动RAG对话服务")
        print("=" * 60)
        
        # 获取用户偏好设置
        self._get_user_preferences()
        
        print("💡 使用说明：")
        print("  - 直接输入问题，系统会检索相关文档并生成回答")
        print("  - 输入 'exit' 或 'quit' 退出服务")
        print("  - 输入 'help' 查看帮助信息")
        print("  - 输入 'status' 查看系统状态")
        print("  - 输入 'settings' 重新设置偏好")
        print("  - 输入 'preferences' 查看当前偏好设置")
        print("  - 输入 'history' 查看对话历史")
        print("=" * 60)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 用户：").strip()
                
                # 检查退出命令
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("👋 感谢使用RAG对话服务，再见！")
                    break
                
                # 检查帮助命令
                if user_input.lower() in ['help', '帮助']:
                    self._show_help()
                    continue
                
                # 检查状态命令
                if user_input.lower() in ['status', '状态']:
                    self._show_status()
                    continue
                
                # 检查设置命令
                if user_input.lower() in ['settings', '设置', '偏好']:
                    self._get_user_preferences()
                    continue
                
                # 检查偏好命令
                if user_input.lower() in ['preferences', '偏好', '当前设置']:
                    self._show_current_preferences()
                    continue
                
                # 检查历史记录命令
                if user_input.lower() in ['history', '历史', '对话记录']:
                    self._show_conversation_history()
                    continue
                
                # 空输入处理
                if not user_input:
                    print("⚠️ 请输入您的问题")
                    continue
                
                # 处理用户问题
                print(f"🔍 正在处理您的问题：{user_input}")
                response = self.chat_with_rag(
                    user_input, 
                    k=5, 
                    use_hybrid_retrieval=self.use_hybrid_retrieval,
                    show_documents=self.show_documents
                )
                
                # 显示回答
                print(f"\n🤖 助手：{response}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ 检测到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"❌ 发生错误：{e}")
                continue

    def _show_help(self):
        """显示帮助信息"""
        print("\n📖 帮助信息")
        print("-" * 40)
        print("💡 系统功能：")
        print("  - 基于医学文档的智能问答")
        print("  - 支持多种检索方式（语义、混合、多跳）")
        print("  - 自动构建知识连接")
        print("\n🔧 可用命令：")
        print("  - exit/quit/退出：退出服务")
        print("  - help/帮助：显示此帮助")
        print("  - status/状态：显示系统状态")
        print("  - settings/设置：重新设置偏好")
        print("  - preferences/偏好：查看当前偏好设置")
        print("  - history/历史：查看对话历史")
        print("\n💬 使用建议：")
        print("  - 问题要具体明确")
        print("  - 可以询问诊断、治疗、药物等相关问题")
        print("  - 系统会自动检索最相关的文档信息")

    def _show_status(self):
        """显示系统状态"""
        print("\n📊 系统状态")
        print("-" * 40)
        
        # 数据库状态
        try:
            doc_count = self.rag_system.qdrant.count()
            print(f"📚 数据库文档数：{doc_count}")
        except Exception as e:
            print(f"📚 数据库状态：无法获取 ({e})")
        
        # 模型状态
        if self.llm:
            print("🤖 LLM模型：已加载")
        else:
            print("🤖 LLM模型：未加载")
        
        # 检索状态
        print("🔍 检索方式：语义 + 混合 + 多跳")
        print("🕸️ 知识连接：已启用")
        print(f"🔀 当前检索：{'混合检索' if self.use_hybrid_retrieval else '多跳检索'}")
        print(f"📚 文档显示：{'启用' if self.show_documents else '禁用'}")
        print(f"💾 历史记录：{len(self.conversation_history)}/{self.max_history}轮")

    def _get_user_preferences(self):
        """获取用户偏好设置"""
        print("\n🔧 请设置您的偏好选项：")
        
        # 询问检索方式
        print("\n1️⃣ 检索方式选择：")
        print("   - 混合检索：结合语义搜索和关键词搜索，通常更准确")
        print("   - 多跳检索：通过多个步骤推理找到答案，适合复杂问题")
        
        while True:
            retrieval_choice = input("请选择检索方式 (输入 '1' 选择混合检索，输入 '2' 选择多跳检索)：").strip()
            if retrieval_choice == "1":
                self.use_hybrid_retrieval = True
                print("✅ 已选择：混合检索")
                break
            elif retrieval_choice == "2":
                self.use_hybrid_retrieval = False
                print("✅ 已选择：多跳检索")
                break
            else:
                print("❌ 无效选择，请输入 '1' 或 '2'")
        
        # 询问是否显示文档内容
        print("\n2️⃣ 文档显示选择：")
        print("   - 显示文档：在回答后显示检索到的原始文档内容")
        print("   - 隐藏文档：只显示生成的回答，不显示原始文档")
        
        while True:
            doc_choice = input("是否需要显示检索到的文档内容？(输入 'y' 显示，输入 'n' 隐藏)：").strip().lower()
            if doc_choice in ['y', 'yes', '是', '显示']:
                self.show_documents = True
                print("✅ 已选择：显示文档内容")
                break
            elif doc_choice in ['n', 'no', '否', '隐藏']:
                self.show_documents = False
                print("✅ 已选择：隐藏文档内容")
                break
            else:
                print("❌ 无效选择，请输入 'y' 或 'n'")
        
        print(f"\n🎯 偏好设置完成！")
        print(f"   - 检索方式：{'混合检索' if self.use_hybrid_retrieval else '多跳检索'}")
        print(f"   - 文档显示：{'显示' if self.show_documents else '隐藏'}")
        print("💡 您可以随时输入 'settings' 重新设置偏好")

    def _format_documents_for_display(self, docs: List[Document]) -> str:
        """格式化文档内容以便显示"""
        if not docs:
            return "没有文档信息可显示。"
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            folder = doc.metadata.get('folder', '')
            
            doc_info = f"【文档{i}】来源：{source}"
            if folder:
                doc_info += f" | 分类：{folder}"
            
            # 截断过长的内容
            if len(content) > 400:
                content = content[:400] + "..."
            
            formatted_docs.append(f"{doc_info}\n{content}\n")
        
        return "\n".join(formatted_docs)

    def _manage_history(self, user_query: str, response: str):
        """
        管理对话历史记录
        
        Args:
            user_query: 用户问题
            response: 模型回答
        """
        # 添加新的对话轮次
        self.conversation_history.append({
            'user': user_query,
            'assistant': response,
            'timestamp': len(self.conversation_history) + 1
        })
        
        # 如果超过最大历史记录，移除最早的记录
        if len(self.conversation_history) > self.max_history:
            removed = self.conversation_history.pop(0)
            print(f"💾 历史记录已满，移除第{removed['timestamp']}轮对话")
    
    def _get_context_from_history(self) -> str:
        """
        从历史记录构建上下文（仅保留最近的几轮）
        
        Returns:
            格式化的历史上下文
        """
        if not self.conversation_history:
            return ""
        
        # 只保留最近3轮对话作为上下文（节省显存）
        recent_history = self.conversation_history[-3:]
        context_parts = []
        
        for item in recent_history:
            context_parts.append(f"用户：{item['user']}")
            context_parts.append(f"助手：{item['assistant']}")
        
        return "\n".join(context_parts)

    def _show_current_preferences(self):
        """显示当前偏好设置"""
        print("\n🔧 当前偏好设置")
        print("-" * 40)
        print(f"🔍 检索方式：{'混合检索' if self.use_hybrid_retrieval else '多跳检索'}")
        print(f"📚 文档显示：{'显示' if self.show_documents else '隐藏'}")
        print("\n💡 说明：")
        if self.use_hybrid_retrieval:
            print("   - 混合检索：结合语义搜索和关键词搜索，提供更准确的检索结果")
        else:
            print("   - 多跳检索：通过多步推理找到答案，适合复杂的医学问题")
        
        if self.show_documents:
            print("   - 文档显示：在回答后会显示检索到的原始文档内容，便于验证")
        else:
            print("   - 文档隐藏：只显示生成的回答，界面更简洁")
        
        print("\n🔄 如需修改，请输入 'settings' 重新设置")

    def _show_conversation_history(self):
        """显示对话历史记录"""
        if not self.conversation_history:
            print("\n📝 暂无对话历史记录")
            return
        
        print(f"\n📝 对话历史记录 (共{len(self.conversation_history)}轮)")
        print("-" * 50)
        
        for i, item in enumerate(self.conversation_history, 1):
            print(f"第{i}轮：")
            print(f"👤 用户：{item['user'][:100]}{'...' if len(item['user']) > 100 else ''}")
            print(f"🤖 助手：{item['assistant'][:100]}{'...' if len(item['assistant']) > 100 else ''}")
            print("-" * 30)
        
        print(f"💾 历史记录限制：{self.max_history}轮")
        print("💡 提示：历史记录会自动管理，超出限制时会移除最早的对话")


def create_chat_interface(rag_system):
    """
    创建对话接口的工厂函数
    
    Args:
        rag_system: RAG系统实例
        
    Returns:
        ChatInterface实例
    """
    return ChatInterface(rag_system)
