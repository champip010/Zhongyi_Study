
# run_complete_evaluation.py
import os
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import asyncio
import json
import time
import numpy as np
import torch
sys.path.append('./src')

from vllm import LLM, SamplingParams
from data_processor import SimpleMedicalProcessor
from advanced_rag_system import AdvancedMedicalRAG
from advanced_evaluator import AdvancedRAGEvaluator

class CompleteRAGBenchmark:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.test_cases = [
            {
                'query': '失眠的中医证候分型有哪些？每种的主要表现是什么？',
                'expected_keywords': ['肝气郁结', '心脾两虚', '心肾不交', '阴虚火旺', '脾虚湿阻'],
                'expected_elements': ['肝气郁结', '心脾两虚', '心肾不交', '阴虚火旺', '脾虚湿阻'],
                'difficulty': 'medium',
                'type': 'enumeration'
            },
            {
                'query': '帕金森病睡眠障碍的诊断标准是什么？具体包括哪些症状？', 
                'expected_keywords': ['运动减少', '肌肉僵直', '静止性震颤', '失眠', '诊断标准'],
                'expected_elements': ['运动减少', '肌肉僵直', '失眠', '诊断标准'],
                'difficulty': 'hard',
                'type': 'diagnostic'
            },
            {
                'query': '镇静催眠药物戒断综合征有哪些临床表现？如何处理？',
                'expected_keywords': ['戒断综合征', '失眠', '焦虑', '震颤', '恶心', '处理'],
                'expected_elements': ['失眠', '焦虑', '震颤', '恶心', '呕吐', '处理方法'],
                'difficulty': 'hard',
                'type': 'comprehensive'
            },
            {
                'query': '睡眠质量下降的推拿疗法具体怎么操作？有哪些注意事项？',
                'expected_keywords': ['推拿', '穴位', '操作', '印堂', '太阳', '注意事项'],
                'expected_elements': ['推拿手法', '穴位选择', '操作步骤', '注意事项'],
                'difficulty': 'hard',
                'type': 'procedural'
            },
            {
                'query': 'PSQI睡眠质量评分的具体标准是什么？如何计算？',
                'expected_keywords': ['PSQI', '评分', '7分', '睡眠质量', '计算', '标准'],
                'expected_elements': ['PSQI', '评分标准', '计算方法', '7分'],
                'difficulty': 'medium',
                'type': 'technical'
            },
            {
                'query': '女性压力性尿失禁的中医辨证分型有哪些？各型的主要症状与治法要点是什么？',
                'expected_keywords': ['肾虚下陷证', '中气下陷证', '脾肾两虚证', '肝郁脾虚证', '辨证分型'],
                'expected_elements': ['肾虚下陷证', '中气下陷证', '脾肾两虚证', '肝郁脾虚证'],
                'difficulty': 'medium',
                'type': 'enumeration'
            },

            {
                'query': '小儿推拿疗法的标准化操作流程具体包括哪些环节？补泻原则和常见禁忌是什么？',
                'expected_keywords': ['施术前准备', '施术方法', '操作顺序', '补泻', '施术后处理', '注意事项', '禁忌'],
                'expected_elements': ['施术前准备', '施术方法', '操作顺序与补泻', '施术后处理', '注意事项', '禁忌'],
                'difficulty': 'hard',
                'type': 'procedural'
            },
            

            {
                'query': '中药药浴的局部药浴与全身药浴应如何规范操作？水温、时间与环境参数各是多少？',
                'expected_keywords': ['局部药浴', '全身药浴', '水温35℃～42℃', '每次15～20分钟', '每日1～2次', '室温约25℃', '相对湿度60%～70%'],
                'expected_elements': ['局部药浴步骤', '全身药浴步骤', '水温范围', '单次时间', '频次', '环境温湿度'],
                'difficulty': 'hard',
                'type': 'procedural'
            },
            

            {
                'query': '高尿酸血症与痛风的饮食指导要点是什么？诊断阈值与需要避免的食物有哪些？',
                'expected_keywords': ['医学营养治疗', '诊断标准', '男性>420 μmol/L', '女性>360 μmol/L', '动物内脏', '贝类', '酒精'],
                'expected_elements': ['诊断阈值', '总体原则', '需避免食物', '随访与监测'],
                'difficulty': 'medium',
                'type': 'technical'
            },
            

            {
                'query': '电针/经皮穴位电刺激（EA/TEAS）在生殖医学中的适应证与参数如何选择？请给出频率、波宽与强度范围及常用疗程。',
                'expected_keywords': ['EA', 'TEAS', '频率2～100Hz', '波宽0.2～0.6ms', '感觉阈3～5mA', '治疗强度6～10mA', '每次20～30min'],
                'expected_elements': ['适应证示例', '频率范围', '波宽范围', '强度设定', '单次时长', '疗程与次数'],
                'difficulty': 'hard',
                'type': 'technical'
            },
            
            

            {
                'query': '耳穴电刺激治疗抑郁症的临床要点是什么？包括适应证、穴位选择、刺激参数、治疗时机/疗程与安全性。',
                'expected_keywords': ['耳穴', '电刺激', '抑郁症', '刺激参数', '治疗时机', '疗程', '安全性'],
                'expected_elements': ['适应证', '穴位/命名', '参数设定', '时机与疗程', '联合用药', '安全性'],
                'difficulty': 'hard',
                'type': 'comprehensive'
            },
            

            {
                'query': '皮秒激光的工作机制与主要适应证有哪些？与调Q激光相比有什么特点？',
                'expected_keywords': ['皮秒激光', 'LIOB（光击穿）', '755/532/1064nm', '文身', '雀斑/日光性黑子', '太田痣', '点阵模式'],
                'expected_elements': ['机制/LIOB', '主波长', '表皮/真皮色素病变适应证', '与调Q差异', '治疗要点'],
                'difficulty': 'hard',
                'type': 'technical'
            },
            

            {
                'query': '司徒氏针挑技术的定义、器具特点、基本操作步骤与禁忌有哪些？',
                'expected_keywords': ['司徒氏针挑', '挑治针', '针尖弯折呈钩状', '皮下白色纤维样组织', '操作步骤', '注意事项', '禁忌'],
                'expected_elements': ['术语定义', '器具要点', '关键手法/步骤', '注意与禁忌'],
                'difficulty': 'hard',
                'type': 'procedural'
            },
            

            {
                'query': '幼年特发性关节炎（JIA）的中西医结合诊疗目标是什么？不同阶段的综合干预方案如何制定？',
                'expected_keywords': ['JIA', '中西医结合', '治疗目标', '缓解炎症', '减少致残率', '方剂/中成药', '外治法', '护理与调摄'],
                'expected_elements': ['治疗目标', '阶段化方案', '方药与中成药', '外治法', '疗效评价', '护理调摄'],
                'difficulty': 'hard',
                'type': 'comprehensive'
            },
            

            {
                'query': '糖尿病足的分级体系有哪些？中医证候分类怎么划分？相应的中西医结合治疗策略是什么？',
                'expected_keywords': ['Wagner分级', 'TEXAS分级', 'IDSA/IWGDF感染分级', '湿热毒盛', '热毒伤阴瘀阻脉络', '气血两虚瘀阻', '综合治疗'],
                'expected_elements': ['分级方法', '中医证候分类', '诊断要点', '中西医结合治疗'],
                'difficulty': 'hard',
                'type': 'diagnostic'
            },
            
            

            {
                'query': '糖尿病足未溃与感染阶段的中医外治法有哪些选择？请给出典型处方/用法与证据要点。',
                'expected_keywords': ['中药足浴', '银黄洗剂', '脉络疏通颗粒外洗', '艾灸', '创面修复', '疗效证据'],
                'expected_elements': ['适应证分层', '外治法名称', '用法与频次', '疗效指标/证据'],
                'difficulty': 'hard',
                'type': 'comprehensive'
            },
        ]
        
        # 初始化高级评估器
        self.advanced_evaluator = AdvancedRAGEvaluator(llm_model)
    
    async def run_complete_benchmark(self, rag_system):
        """运行完整的评估基准测试"""
        
        print("🚀 启动完整RAG评估系统")
        print("="*80)
        print("📋 评估维度:")
        print("   📊 基础指标: 关键词覆盖、响应时间、成功率")
        print("   🧠 LLM评判: 忠实度、相关性、医学正确性、完整性")
        print("="*80)
        
        # 测试的RAG策略
            #  'basic_dense': '基础Dense检索',
            # 'hyde_enhanced': 'HyDE增强检索', 
        strategies = {
            'basic_dense': '基础Dense检索',
            'hyde_enhanced': 'HyDE增强检索',
            'multi_hop': '多跳检索',
            'self_rag': 'Self-RAG'
        }
        
        all_results = {}
        
        for strategy_name, strategy_desc in strategies.items():
            print(f"\n📊 评估策略: {strategy_desc}")
            print("-" * 50)
            
            # 1. 基础评估
            basic_results = await self._run_basic_evaluation(rag_system, strategy_name)
            
            # 2. 高级LLM评估
            print("   🧠 运行LLM高级评估...")
            advanced_results = await self._run_advanced_evaluation(rag_system, strategy_name)
            
            # 3. 合并结果
            all_results[strategy_name] = {
                'description': strategy_desc,
                'basic_metrics': basic_results,
                'advanced_metrics': advanced_results,
                'combined_score': self._calculate_combined_score(basic_results, advanced_results)
            }
            
            print(f"   ✅ {strategy_desc} 评估完成")
        
        # 生成综合报告
        self._generate_complete_report(all_results)
        
        # 保存详细结果
        with open('complete_rag_evaluation.json', 'w', encoding='utf-8') as f:
            try:
                # 创建简化的可序列化版本
                simplified_results = {}
                for strategy, data in all_results.items():
                    simplified_results[strategy] = {
                        'description': data['description'],
                        'combined_score': float(data['combined_score']['total_score']),
                        'basic_metrics': {
                            k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in data['basic_metrics']['aggregate_stats'].items()
                        },
                        'advanced_metrics': {
                            k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in data['advanced_metrics']['aggregate_metrics'].items()
                        }
                    }
                
                json.dump(simplified_results, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"保存详细结果失败: {e}")
                # 至少保存评分
                simple_scores = {
                    strategy: {
                        'description': data['description'],
                        'score': float(data['combined_score']['total_score'])
                    }
                    for strategy, data in all_results.items()
                }
                json.dump(simple_scores, f, ensure_ascii=False, indent=2)
        
        return all_results
    
    async def _run_basic_evaluation(self, rag_system, strategy_name):
        """运行基础评估"""
        
        print("   📊 运行基础指标评估...")
        
        strategy_results = []
        total_time = 0
        
        for test_case in self.test_cases:
            query = test_case['query']
            expected_keywords = test_case['expected_keywords']
            
            # 执行查询并计时
            start_time = time.time()
            try:
                result = self._execute_strategy(rag_system, strategy_name, query)
                execution_time = time.time() - start_time
                total_time += execution_time
                
                # 基础评估
                scores = self._evaluate_basic_metrics(
                    query, result['answer'], expected_keywords
                )
                
                strategy_results.append({
                    'query': query,
                    'answer': result['answer'],
                    'execution_time': execution_time,
                    'scores': scores,
                    'difficulty': test_case['difficulty'],
                    'success': True
                })
                
            except Exception as e:
                strategy_results.append({
                    'query': query,
                    'answer': f'Error: {str(e)}',
                    'execution_time': 0,
                    'scores': {'keyword_coverage': 0, 'answer_length': 0, 'relevance': 0},
                    'difficulty': test_case['difficulty'],
                    'success': False
                })
        
        # 计算基础统计
        stats = self._calculate_basic_stats(strategy_results)
        
        return {
            'individual_results': strategy_results,
            'aggregate_stats': stats,
            'total_execution_time': total_time
        }
    
    async def _run_advanced_evaluation(self, rag_system, strategy_name):
        """运行高级LLM评估"""
        
        # 准备测试用例（转换格式以适配高级评估器）
        advanced_test_cases = []
        for test_case in self.test_cases:
            advanced_test_cases.append({
                'query': test_case['query'],
                'expected_elements': test_case['expected_elements']
            })
        
        # 创建一个适配器来使用不同策略
        class StrategyAdapter:
            def __init__(self, base_system, strategy):
                self.base_system = base_system
                self.strategy = strategy
            
            def self_rag_query(self, query):
                if self.strategy == 'self_rag':
                    return self.base_system.self_rag_query(query)
                else:
                    # 对其他策略进行适配
                    if self.strategy == 'basic_dense':
                        docs = self.base_system._dense_retrieval(query, top_k=5)
                    elif self.strategy == 'hyde_enhanced':
                        enhanced_query = self.base_system.hyde_enhanced_query(query)
                        docs = self.base_system._dense_retrieval(enhanced_query, top_k=5)
                    elif self.strategy == 'multi_hop':
                        docs = self.base_system.multi_hop_retrieval(query, max_hops=2)
                    else:
                        docs = self.base_system._dense_retrieval(query, top_k=5)
                    
                    answer = self.base_system._generate_rag_answer(query, docs)
                    
                    return {
                        'query': query,
                        'answer': answer,
                        'method': self.strategy,
                        'retrieval_used': True,
                        'retrieved_docs': docs
                    }
        
        adapted_system = StrategyAdapter(rag_system, strategy_name)
        
        # 运行高级评估
        advanced_results = await self.advanced_evaluator.evaluate_rag_system(
            adapted_system, advanced_test_cases
        )
        
        return advanced_results
    
    def _execute_strategy(self, rag_system, strategy_name, query):
        """执行特定的RAG策略"""
        
        if strategy_name == 'basic_dense':
            docs = rag_system._dense_retrieval(query, top_k=5)
            answer = rag_system._generate_rag_answer(query, docs)
            return {'answer': answer, 'docs_used': len(docs)}
            
        elif strategy_name == 'hyde_enhanced':
            enhanced_query = rag_system.hyde_enhanced_query(query)
            docs = rag_system._dense_retrieval(enhanced_query, top_k=5)
            answer = rag_system._generate_rag_answer(query, docs)
            return {'answer': answer, 'docs_used': len(docs)}
            
        elif strategy_name == 'multi_hop':
            docs = rag_system.multi_hop_retrieval(query, max_hops=2)
            answer = rag_system._generate_rag_answer(query, docs)
            return {'answer': answer, 'docs_used': len(docs)}
            
        elif strategy_name == 'self_rag':
            result = rag_system.self_rag_query(query)
            return result
            
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _evaluate_basic_metrics(self, query, answer, expected_keywords):
        """基础指标评估"""
        
        # 1. 关键词覆盖率
        keyword_hits = sum(1 for kw in expected_keywords if kw in answer)
        keyword_coverage = keyword_hits / len(expected_keywords) if expected_keywords else 0
        
        # 2. 答案长度
        answer_length = len(answer)
        length_score = min(answer_length / 300, 1.0)  # 300字符为满分
        
        # 3. 词汇相关性
        import jieba
        query_words = set(jieba.cut(query))
        answer_words = set(jieba.cut(answer))
        
        if len(query_words) > 0:
            relevance = len(query_words & answer_words) / len(query_words)
        else:
            relevance = 0
        
        return {
            'keyword_coverage': keyword_coverage,
            'answer_length': answer_length,
            'length_score': length_score,
            'relevance': relevance
        }
    
    def _calculate_basic_stats(self, results):
        """计算基础统计信息"""
        
        valid_results = [r for r in results if r['success']]
        
        if not valid_results:
            return {'success_rate': 0}
        
        return {
            'success_rate': len(valid_results) / len(results),
            'avg_keyword_coverage': np.mean([r['scores']['keyword_coverage'] for r in valid_results]),
            'avg_length_score': np.mean([r['scores']['length_score'] for r in valid_results]),
            'avg_relevance': np.mean([r['scores']['relevance'] for r in valid_results]),
            'avg_execution_time': np.mean([r['execution_time'] for r in valid_results]),
        }
    
    def _calculate_combined_score(self, basic_results, advanced_results):
        """计算综合得分"""
        
        basic_stats = basic_results['aggregate_stats']
        
        # 从高级评估结果中提取平均分数
        individual_scores = advanced_results.get('individual_scores', [])
        
        if not individual_scores:
            return basic_stats.get('avg_keyword_coverage', 0) * 0.5
        
        # 计算高级指标平均值
        advanced_scores = {}
        score_keys = ['faithfulness', 'answer_relevancy', 'medical_correctness', 'completeness']
        
        for key in score_keys:
            scores = []
            for item in individual_scores:
                if key in item.get('scores', {}):
                    scores.append(item['scores'][key])
            advanced_scores[key] = np.mean(scores) if scores else 0.5
        
        # 综合评分（权重可调整）
        combined_score = (
            basic_stats.get('avg_keyword_coverage', 0) * 0.2 +      # 基础-关键词覆盖 20%
            basic_stats.get('avg_relevance', 0) * 0.1 +             # 基础-相关性 10%
            advanced_scores.get('faithfulness', 0.5) * 0.25 +       # 高级-忠实度 25%
            advanced_scores.get('answer_relevancy', 0.5) * 0.15 +   # 高级-相关性 15%
            advanced_scores.get('medical_correctness', 0.5) * 0.25 + # 高级-医学正确性 25%
            advanced_scores.get('completeness', 0.5) * 0.05         # 高级-完整性 5%
        )
        
        return {
            'total_score': combined_score,
            'basic_component': basic_stats.get('avg_keyword_coverage', 0) * 0.3 + basic_stats.get('avg_relevance', 0) * 0.1,
            'advanced_component': sum(advanced_scores.values()) / len(advanced_scores) * 0.7 if advanced_scores else 0,
            'individual_advanced_scores': advanced_scores
        }
    
    def _generate_complete_report(self, all_results):
        """生成完整评估报告"""
        
        print("\n" + "="*100)
        print("📈 完整RAG评估报告")
        print("="*100)
        
        # 1. 综合排名
        print("\n🏆 综合性能排名 (基础指标 + LLM评判):")
        ranking = sorted(all_results.items(), 
                        key=lambda x: x[1]['combined_score']['total_score'], 
                        reverse=True)
        
        for i, (strategy, data) in enumerate(ranking, 1):
            desc = data['description']
            combined = data['combined_score']
            basic = data['basic_metrics']['aggregate_stats']
            
            print(f"\n{i}. {desc}")
            print(f"   🎯 综合得分: {combined['total_score']:.3f}")
            print(f"   📊 基础得分: {combined['basic_component']:.3f}")
            print(f"   🧠 LLM得分: {combined['advanced_component']:.3f}")
            print(f"   ⚡ 平均耗时: {basic.get('avg_execution_time', 0):.2f}s")
        
        # 2. 详细对比表
        print(f"\n📊 详细评估对比:")
        print("-" * 120)
        print(f"{'策略':<15} {'综合得分':<8} {'关键词覆盖':<10} {'忠实度':<8} {'相关性':<8} {'医学正确性':<10} {'平均耗时':<8}")
        print("-" * 120)
        
        for strategy, data in all_results.items():
            desc = data['description']
            combined = data['combined_score']
            basic = data['basic_metrics']['aggregate_stats']
            advanced_scores = combined['individual_advanced_scores']
            
            print(f"{desc:<15} "
                  f"{combined['total_score']:<8.3f} "
                  f"{basic.get('avg_keyword_coverage', 0):<10.3f} "
                  f"{advanced_scores.get('faithfulness', 0):<8.3f} "
                  f"{advanced_scores.get('answer_relevancy', 0):<8.3f} "
                  f"{advanced_scores.get('medical_correctness', 0):<10.3f} "
                  f"{basic.get('avg_execution_time', 0):<8.2f}s")
        
        # 3. 分类分析
        print(f"\n📈 各维度最佳表现:")
        
        # 找出各维度最佳
        best_overall = max(all_results.items(), key=lambda x: x[1]['combined_score']['total_score'])
        best_speed = min(all_results.items(), key=lambda x: x[1]['basic_metrics']['aggregate_stats'].get('avg_execution_time', float('inf')))
        best_medical = max(all_results.items(), key=lambda x: x[1]['combined_score']['individual_advanced_scores'].get('medical_correctness', 0))
        best_faithfulness = max(all_results.items(), key=lambda x: x[1]['combined_score']['individual_advanced_scores'].get('faithfulness', 0))
        
        print(f"   🥇 综合最佳: {best_overall[1]['description']} ({best_overall[1]['combined_score']['total_score']:.3f})")
        print(f"   ⚡ 速度最快: {best_speed[1]['description']} ({best_speed[1]['basic_metrics']['aggregate_stats'].get('avg_execution_time', 0):.2f}s)")
        print(f"   🏥 医学准确性最高: {best_medical[1]['description']} ({best_medical[1]['combined_score']['individual_advanced_scores'].get('medical_correctness', 0):.3f})")
        print(f"   📖 忠实度最高: {best_faithfulness[1]['description']} ({best_faithfulness[1]['combined_score']['individual_advanced_scores'].get('faithfulness', 0):.3f})")
        
        # 4. 使用建议
        print(f"\n💡 使用建议:")
        
        best_strategy_name = ranking[0][0]
        best_strategy_desc = ranking[0][1]['description']
        
        print(f"   🎯 推荐策略: {best_strategy_desc}")
        print(f"   📋 适用场景:")
        
        if best_strategy_name == 'self_rag':
            print(f"      • 需要高质量、准确的医学问答")
            print(f"      • 对响应时间要求不是极其严格")
            print(f"      • 重视答案的专业性和可靠性")
        elif best_strategy_name == 'multi_hop':
            print(f"      • 需要全面、深入的信息整合")
            print(f"      • 复杂的多方面医学问题")
        elif best_strategy_name == 'hyde_enhanced':
            print(f"      • 查询理解困难的情况")
            print(f"      • 需要语义增强的检索")
        else:
            print(f"      • 对响应速度要求很高的场景")
            print(f"      • 简单直接的信息查询")
        
        print(f"\n   ⚠️  注意事项:")
        print(f"      • 医学信息仅供参考，请咨询专业医师")
        print(f"      • 定期评估和更新知识库")
        print(f"      • 根据具体应用场景选择合适策略")

async def main():
    # torch.cuda.memory._record_memory_history(max_entries=100000)

    """主函数"""
    print("🚀 启动完整RAG评估系统")
    local_model_path = '../../llamaindex+Qwen-7b/model/Qwen/Qwen-7B-Chat'

    # 1. 加载模型
    # Qwen-7B-Chat的attentionhead为28，用vllm只能设置能除尽28的数
    print("\n1. 加载模型...")
    llm = LLM(
        model="Qwen/Qwen2-VL-7B-Instruct",
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )
    print("✅ 模型加载完成")

    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"


    # 2. 初始化RAG系统
    print("\n2. 构建先进RAG系统...")
    config = {
        'chunk_size': 600,
        'chunk_overlap': 150,
        "reset_chroma" : False,  # 是否重置Chroma数据库
        "update_chroma_knowledge_connection": False,  # 是否更新Chroma知识库
    }
    rag_system = AdvancedMedicalRAG(llm, config)

    if rag_system.chroma.count() == 0:

        # 2.2 处理数据
        print("\n2.1 处理数据...")
        processor = SimpleMedicalProcessor()
        
        # 查找数据路径
        data_paths = ['../../processed_md/']
        documents = None
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    documents = processor.load_documents(path)
                    print(f"✅ 数据加载完成: {path}")
                    break
                except:
                    continue
        
        if not documents:
            print("❌ 找不到数据文件")
            return
        
        chunks = processor.smart_chunk_documents(documents)
        print(f"✅ 文档分块完成: {len(chunks)} 个chunks")
        # 2.3 构建索引
        print("\n2.2 构建索引...")
        rag_system.build_advanced_index(chunks)

    print("✅ RAG系统构建完成")

    
    # 3. 运行完整评估
    print("\n3. 开始完整评估...")
    benchmark = CompleteRAGBenchmark(llm)
    results = await benchmark.run_complete_benchmark(rag_system)
    

    print(f"\n✅ 完整评估结束！")
    print(f"📄 详细结果已保存到: complete_rag_evaluation.json")
    # 保存数据
    # torch.cuda.memory._dump_snapshot("memory_record.pickle")

    # # 停掉记录，关闭snapshot
    # torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    asyncio.run(main())