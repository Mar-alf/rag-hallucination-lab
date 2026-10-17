from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from test_questions import TEST_QUESTIONS

load_dotenv()

# 读取材料文档并切分
def load_and_split(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("# ") if c.strip()]
    return chunks

# 向量化并存入 ChromaDB
def build_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    client = chromadb.Client()
    collection = client.create_collection(name="materials_baseline")
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(ids=[str(i)], documents=[chunk], embeddings=[emb.tolist()])
    return collection, model

#  检索
def search(query, collection, model, top_k=3):
    query_emb = model.encode([query])
    results = collection.query(query_emb.tolist(), n_results=top_k)
    return results['documents'][0]

# 基线版 Prompt（无 Few-shot，无事实约束）
def build_prompt_baseline(question, retrieved_docs):
    context = "\n".join(retrieved_docs)
    return f"根据以下材料回答问题：\n{context}\n\n问题：{question}\n回答："


if __name__ == "__main__":
    chunks = load_and_split("data/materials.md")
    print(f"加载了 {len(chunks)} 条材料说明")

    collection, model = build_index(chunks)
    print("索引构建完成")

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    results = []
    for i, query in enumerate(TEST_QUESTIONS):
        docs = search(query, collection, model)
        prompt = build_prompt_baseline(query, docs)
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"[ERROR] {e}"
        results.append({"question": query, "answer": answer, "docs": docs})
        print(f"[{i+1}/50] {query}")
        print(f"回答: {answer[:100]}...")

    with open("results_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("完成，结果保存到 results_baseline.json")
