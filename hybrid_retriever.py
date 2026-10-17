from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

#  初始化重排序模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def bm25_search(query, chunks, top_k=50):
    tokenized = [list(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(chunks[i], scores[i]) for i in ranked[:top_k]]

def vector_search(query, collection, model, top_k=50):
  # 向量语义检索
    query_emb = model.encode([query])
    results = collection.query(query_emb.tolist(), n_results=top_k)
    return results['documents'][0]

def rrf_fusion(bm25_results, vector_results, k=60):
    # RRF 倒数排名融合
    scores = {}
    for rank, (doc, _) in enumerate(bm25_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(vector_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

def rerank(query, candidates, top_k=5):
    # 用 Cross-Encoder 重排序，保留 Top-5
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def hybrid_search(query, chunks, collection, model):
    # 完整流程：BM25 + 向量 → RRF → 重排序 → Top-5
    bm25_results = bm25_search(query, chunks, top_k=50)
    vector_results = vector_search(query, collection, model, top_k=50)
    fused = rrf_fusion(bm25_results, vector_results)
    final = rerank(query, fused[:50], top_k=5)
    return final
