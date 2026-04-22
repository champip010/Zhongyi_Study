# src/advanced_evaluator.py (完整修复版)
import asyncio
from typing import List, Dict, Any
import numpy as np
import json
import jieba

class AdvancedRAGEvaluator:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.evaluation_criteria = {
            'faithfulness': '答案是否忠实于参考文档',
            'answer_relevancy': '答案与问题的相关性',  
            'context_precision': '检索上下文的精确度',
            'context_recall': '检索上下文的召回率',
            'answer_correctness': '答案的医学正确性',
            'answer_completeness': '答案的完整性'
        }
    
    async def evaluate_rag_system(self, rag_system, test_cases: List[Dict]) -> Dict[str, Any]:
        """全面评估RAG系统"""
        
        results = {
            'individual_scores': [],
            'aggregate_metrics': {},
            'detailed_analysis': {}
        }
        
        for test_case in test_cases:
            query = test_case['query']
            expected_elements = test_case.get('expected_elements', [])
            
            # 获取RAG响应
            try:
                response = rag_system.self_rag_query(query)
                
                # 多维度评估
                scores = await self._comprehensive_evaluation(
                    query, response, expected_elements
                )
                
                results['individual_scores'].append({
                    'query': query,
                    'response': response,
                    'scores': scores
                })
                
            except Exception as e:
                print(f"⚠️ 评估查询失败: {query[:30]}... - {str(e)}")
                # 添加默认分数以避免中断
                results['individual_scores'].append({
                    'query': query,
                    'response': {'answer': f'Error: {str(e)}', 'retrieval_used': False},
                    'scores': {
                        'faithfulness': 0.0,
                        'answer_relevancy': 0.0,
                        'medical_correctness': 0.0,
                        'completeness': 0.0,
                        'context_precision': 0.0
                    }
                })
        
        # 计算聚合指标
        results['aggregate_metrics'] = self._calculate_aggregate_metrics(
            results['individual_scores']
        )
        
        return results
    
    async def _comprehensive_evaluation(self, query: str, response: Dict, expected_elements: List[str]) -> Dict[str, float]:
        """综合评估单个响应"""
        
        scores = {}
        
        # 1. Faithfulness - 答案是否忠实于文档
        if response.get('retrieval_used', False):
            retrieved_docs = response.get('retrieved_docs', [])
            scores['faithfulness'] = await self._evaluate_faithfulness(
                response['answer'], 
                retrieved_docs
            )
            
            # 5. Context Quality (如果使用了检索)
            scores['context_precision'] = await self._evaluate_context_precision(
                query, retrieved_docs
            )
        else:
            scores['faithfulness'] = 0.7  # 直接生成的默认分数
            scores['context_precision'] = 0.0  # 没有检索就没有上下文精度
        
        # 2. Answer Relevancy - 答案相关性
        scores['answer_relevancy'] = await self._evaluate_relevancy(
            query, response['answer']
        )
        
        # 3. Medical Correctness - 医学正确性
        scores['medical_correctness'] = await self._evaluate_medical_correctness(
            query, response['answer']
        )
        
        # 4. Completeness - 完整性
        scores['completeness'] = self._evaluate_completeness(
            response['answer'], expected_elements
        )
        
        return scores
    
    async def _evaluate_faithfulness(self, answer: str, retrieved_docs: List) -> float:
        """评估答案忠实度"""
        
        if not retrieved_docs:
            return 0.5
        
        # 处理不同类型的文档对象
        context_parts = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                context_parts.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                context_parts.append(doc['page_content'])
            elif isinstance(doc, str):
                context_parts.append(doc)
            else:
                context_parts.append(str(doc))
        
        context = " ".join(context_parts)
        
        faithfulness_prompt = f"""评估答案是否忠实于提供的参考文档。

参考文档：
{context[:1500]}...

答案：
{answer}

请评估答案中的信息是否都能在参考文档中找到支持。
评分标准：
1.0 - 答案完全基于参考文档，无额外信息
0.8 - 答案主要基于参考文档，少量合理推理
0.6 - 答案部分基于参考文档，有一些额外信息
0.4 - 答案少量基于参考文档，较多额外信息  
0.2 - 答案基本不基于参考文档
0.0 - 答案与参考文档矛盾

请只回答一个数字（0.0-1.0）："""
        
        return await self._get_llm_score(faithfulness_prompt)
    
    async def _evaluate_relevancy(self, query: str, answer: str) -> float:
        """评估答案相关性"""
        
        relevancy_prompt = f"""评估答案与问题的相关性。

问题：{query}
答案：{answer}

评分标准：
1.0 - 答案完美回答了问题
0.8 - 答案很好地回答了问题
0.6 - 答案基本回答了问题
0.4 - 答案部分回答了问题
0.2 - 答案勉强涉及了问题
0.0 - 答案完全不相关

请只回答一个数字（0.0-1.0）："""
        
        return await self._get_llm_score(relevancy_prompt)
    
    async def _evaluate_medical_correctness(self, query: str, answer: str) -> float:
        """评估医学正确性"""
        
        correctness_prompt = f"""作为医学专家，评估以下医学问答的专业正确性。

问题：{query}
答案：{answer}

评分标准：
1.0 - 医学信息完全正确，专业术语准确
0.8 - 医学信息基本正确，术语恰当
0.6 - 医学信息大部分正确，有小错误
0.4 - 医学信息部分正确，有明显错误
0.2 - 医学信息大部分错误
0.0 - 医学信息完全错误或有害

请只回答一个数字（0.0-1.0）："""
        
        return await self._get_llm_score(correctness_prompt)
    
    async def _evaluate_context_precision(self, query: str, retrieved_docs: List) -> float:
        """评估检索上下文的精确度"""
        
        if not retrieved_docs:
            return 0.0
        
        # 处理文档内容
        context_parts = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                context_parts.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                context_parts.append(doc['page_content'])
            elif isinstance(doc, str):
                context_parts.append(doc)
            else:
                context_parts.append(str(doc))
        
        context = " ".join(context_parts)
        
        precision_prompt = f"""评估检索到的文档与查询问题的相关性。

查询问题：{query}
检索文档：{context[:1200]}...

评分标准：
1.0 - 所有检索文档都与问题高度相关
0.8 - 大部分文档与问题相关
0.6 - 一半以上文档与问题相关
0.4 - 少部分文档与问题相关
0.2 - 很少文档与问题相关
0.0 - 检索文档与问题无关

请只回答一个数字（0.0-1.0）："""
        
        return await self._get_llm_score(precision_prompt)
    
    def _evaluate_completeness(self, answer: str, expected_elements: List[str]) -> float:
        """评估答案完整性"""
        
        if not expected_elements:
            # 如果没有预期元素，基于答案长度和结构来评估完整性
            answer_length = len(answer)
            if answer_length > 200:
                return 0.8
            elif answer_length > 100:
                return 0.6
            elif answer_length > 50:
                return 0.4
            else:
                return 0.2
            
        covered_elements = sum(1 for element in expected_elements if element in answer)
        return covered_elements / len(expected_elements)
    
    def _calculate_aggregate_metrics(self, individual_scores: List[Dict]) -> Dict[str, float]:
        """计算聚合指标"""
        
        if not individual_scores:
            return {}
        
        # 提取所有分数
        all_scores = {}
        score_keys = ['faithfulness', 'answer_relevancy', 'medical_correctness', 'completeness', 'context_precision']
        
        for key in score_keys:
            scores = []
            for item in individual_scores:
                if key in item.get('scores', {}):
                    score = item['scores'][key]
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        scores.append(score)
            
            if scores:
                all_scores[f'avg_{key}'] = np.mean(scores)
                all_scores[f'std_{key}'] = np.std(scores)
            else:
                all_scores[f'avg_{key}'] = 0.0
                all_scores[f'std_{key}'] = 0.0
        
        # 计算总体评分
        main_scores = [
            all_scores.get('avg_faithfulness', 0),
            all_scores.get('avg_answer_relevancy', 0),
            all_scores.get('avg_medical_correctness', 0),
            all_scores.get('avg_completeness', 0)
        ]
        
        all_scores['overall_score'] = np.mean([s for s in main_scores if s > 0])
        
        return all_scores
    
    async def _get_llm_score(self, prompt: str) -> float:
        """获取LLM评分"""
        
        try:
            from vllm import SamplingParams
            params = SamplingParams(temperature=0.1, max_tokens=20, stop=["<|im_end|>"])
            
            outputs = self.llm.generate([prompt], params)
            score_text = outputs[0].outputs[0].text.strip()
            
            # 提取数字
            import re
            numbers = re.findall(r'0\.\d+|1\.0|^0$|^1$', score_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # 确保在0-1范围内
            else:
                # 如果没找到数字，尝试从文本中推断
                if '完全' in score_text and ('正确' in score_text or '相关' in score_text):
                    return 0.9
                elif '基本' in score_text and ('正确' in score_text or '相关' in score_text):
                    return 0.7
                elif '部分' in score_text:
                    return 0.5
                elif '错误' in score_text or '无关' in score_text:
                    return 0.2
                else:
                    return 0.5  # 默认中等分数
                
        except Exception as e:
            print(f"⚠️ LLM评分失败: {e}")
            return 0.5