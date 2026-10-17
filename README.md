# RAG 幻觉治理与评测优化

一组对照实验：**同一条生成链路，只改检索层与 Prompt，把编造率从 28% 降到 10%，Faithfulness 从 0.72 提到 0.89。**

实验语料是 15 条床垫材料说明，测试集是 50 个人工设计的问题（对比类、适用人群类、保养类、极值类、场景类）。

---

## 1. 问题：单路向量检索 + 裸 Prompt，模型会编

真实翻车例子：

> 用户问：「驼绒适合什么人」
> 检索 top-3 里根本没有驼绒（向量检索把"适合什么人"的语义对到了新疆棉/马尾毛），
> 模型照样答：「驼绒适合怕冷的人」—— **材料说明里没这句，这是编的。**

拆开看是三处独立缺陷：

1. **召回不全**：专有名词（驼绒、酷布9环）在向量空间里不占优势，单路向量检索会漏；
2. **排序太粗**：即使召回对了，双塔模型给的相似度也无法精细判断"这段能不能回答这个问题"；
3. **Prompt 没有退路**：没有 few-shot、没有"无依据就说没提及"的约束，模型就自由发挥。

## 2. 实验设计

| | 基线版 `rag_core_baseline.py` | 优化版 `rag_core_optimized.py` |
|---|---|---|
| 检索 | 向量检索 top-3 | BM25（top-50）+ 向量（top-50）→ **RRF 融合** → **Cross-Encoder 重排** top-5 |
| Prompt | "根据以下材料回答问题" + 上下文 | 3 条 few-shot + 事实约束（"只能使用下面的材料说明，无依据时明确说材料库中未提及"） |
| 上下文管理 | 无 | 滑动窗口 5 轮 + 超窗内容摘要压缩 |
| 生成模型 | deepseek-chat, temperature 0.7 | 同左（**唯一变量只有检索与 Prompt**） |

## 3. 结果

| 指标 | 基线版 | 优化版 |
|---|---|---|
| 50 题中编造回答 | 14 | 5 |
| 幻觉率 | 28% | 10% |
| Faithfulness（RAGAS，DeepSeek 作裁判） | 0.72 | 0.89 |

逐题原始输出在 [`results_baseline.json`](results_baseline.json) / [`results_optimized.json`](results_optimized.json)，每条都带当次检索到的 `docs`，可以直接对照"检索错了"还是"生成错了"。

> 样本量小（50 题），数字只用于同一套题内的相对比较，不当绝对指标看。

## 4. 每一步为什么有效

| 观察到的问题 | 手段 | 为什么管用 |
|---|---|---|
| 「天然新疆棉和驼绒有什么区别」只召回新疆棉 | 加 BM25 关键词检索 | 专有名词走字面匹配，正好补向量检索的短板 |
| 两路分数不可比（余弦距离 vs BM25 分） | RRF（`k=60`）只用**排名**融合 | 不需要调权重，天然抗分数尺度差异 |
| RRF 后 Top-50 仍混着无关文档 | Cross-Encoder 联合编码精排，保留 top-5 | query 和 doc 拼一起算相似度，能判断"这段是否回答问题" |
| Few-shot + 长上下文导致 token 暴涨 | 滑动窗口 5 轮 + 摘要压缩 | 只保留最近交互，超出的压成一句摘要 |

核心代码在 [`hybrid_retriever.py`](hybrid_retriever.py)：`bm25_search` → `vector_search` → `rrf_fusion` → `rerank`，四个函数各自独立、可单独替换。

## 5. 文件结构

| 文件 | 作用 |
|---|---|
| `rag_core_baseline.py` | 基线版：向量检索 + 简单 Prompt |
| `rag_core_optimized.py` | 优化版：混合检索 + 重排 + few-shot |
| `hybrid_retriever.py` | BM25 / 向量 / RRF / Cross-Encoder 重排 |
| `context_manager.py` | few-shot Prompt、滑动窗口 `ConversationHistory`、摘要压缩 |
| `fault_tolerance.py` | `retry_with_timeout` / `classify_error` / `fallback_to_mysql` |
| `mattress_db.py` | MySQL 数据层 + ChromaDB 失败时的模糊查询兜底 |
| `test_questions.py` | 50 个测试问题 |
| `run_ragas.py` | 用 RAGAS 跑 Faithfulness / Answer Relevancy |
| `data/materials.md` | 15 条材料说明（语料） |
| `results_*.json` | 两个版本的逐题输出 |

## 6. 运行

```bash
python -m venv ragas-env
# Windows: ragas-env\Scripts\activate
source ragas-env/bin/activate
pip install -r requirements.txt

cp .env.example .env          # 填 DEEPSEEK_API_KEY

export HF_ENDPOINT=https://hf-mirror.com   # 国内下载模型加速
python rag_core_optimized.py               # 跑 50 题，产出 results_optimized.json
python run_ragas.py                        # 评测打分
```

## 7. 容错设计

链路里任何一环挂掉都不应该让整条问答失败：

- `retry_with_timeout(func, max_retries=3, timeout=10)`：单次查询超时即中断；
- `classify_error`：区分 `timeout` / `connection` / `data` 三类 —— **数据类错误不重试**（重试也不会变对）；
- `fallback_to_mysql`：ChromaDB 检索失败时退回 MySQL 关键词模糊查，宁可降级也别整体报错。

## 8. 局限与下一步

- **样本太小**：15 条语料、50 个问题，方差大；下一步扩到真实商品目录并做分桶统计（对比 / 极值 / 场景）。
- **中文没做专门处理**：`bm25_search` 目前按字符切分（`list(c)`），中文应该换 jieba 分词；embedding 也换 `text2vec-base-chinese` 更合适。
- **幻觉判定还是 LLM-as-judge**：可加 claim-level NLI 校验，把"哪一句编的"定位到句子级。
- **检索与生成耦合**：可以再叠一层 self-RAG（先判断检索内容是否足够，不够就明确拒答）。

## 9. License

[MIT](LICENSE) © 2026 Mar-alf
