from sentence_transformers import SentenceTransformer
import chromadb
from context_manager import build_prompt

# 1. 读取材料文档并切分
def load_and_split(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 按 "# " 切分，每段是一个材料
    chunks = [c.strip() for c in text.split("# ") if c.strip()]
    return chunks

# 2. 向量化并存入 ChromaDB
def build_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    client = chromadb.Client()
    collection = client.create_collection(name="materials")
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(ids=[str(i)], documents=[chunk], embeddings=[emb.tolist()])
    return collection, model

# 3. 检索
def search(query, collection, model, top_k=3):
    query_emb = model.encode([query])
    results = collection.query(query_emb.tolist(), n_results=top_k)
    return results['documents'][0]

if __name__ == "__main__":
    chunks = load_and_split("data/materials.md")
    print(f"加载了 {len(chunks)} 条材料说明")

    collection, model = build_index(chunks)
    print("索引构建完成")

    query = "天然新疆棉和驼绒有什么区别"
    docs = search(query, collection, model)
    print(f"检索到 {len(docs)} 条相关文档")

    prompt = build_prompt(query, docs)
    print("\n构建的 Prompt：")
    print(prompt[:500])
