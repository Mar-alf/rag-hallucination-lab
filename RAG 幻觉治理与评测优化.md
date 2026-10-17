# RAG 幻觉治理与评测优化

通过混合检索、重排序、Few-shot 约束降低 RAG 系统幻觉率。

## 技术栈
Python / ChromaDB / Sentence-Transformers / Rank-BM25 / Cross-Encoder / DeepSeek API

## 实验结果
- 基线版：50 个问题，14 个编造，幻觉率 30%
- 优化版：50 个问题，5 个编造，Faithfulness 0.72 → 0.89

## 文件说明
- rag_core_baseline.py：基线版
- rag_core_optimized.py：优化版
- hybrid_retriever.py：混合检索 + RRF + 重排序
- context_manager.py：Few-shot + 上下文管理
- fault_tolerance.py：容错机制