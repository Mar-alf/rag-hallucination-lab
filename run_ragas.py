import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# 1. 配置 DeepSeek 作为裁判
client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

judge_llm = llm_factory(
    model="deepseek-chat",
    client=client,
    provider="openai"
)

# 2. 加载优化版结果
with open("results_optimized.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 3. 整理成 RAGAS 需要的格式
data = {
    "user_input": [item["question"] for item in results],
    "response": [item["answer"] for item in results],
    "retrieved_contexts": [item["docs"] for item in results]
}

dataset = Dataset.from_dict(data)

# 4. 运行评测
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm
)

# 5. 输出结果
print(result)
