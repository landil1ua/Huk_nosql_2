# scripts/06_hybrid_search.py
import os
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K      = 10   # беремо ширше, щоб RRF міг переранжувати
DISPLAY_K  = 5
RRF_K      = 60   # стандартна константа RRF

pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df    = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# Корпус: title + abstract кожного документа, токенізований за пробілами
corpus_tokens = (
    (df["title"].fillna("") + " " + df["abstract"].fillna(""))
    .str.lower()
    .str.split()
    .tolist()
)
bm25 = BM25Okapi(corpus_tokens)

# Швидкий lookup: arxiv_id -> рядок df (для відображення результатів)
id_to_row: dict[str, pd.Series] = {str(row["id"]): row for _, row in df.iterrows()}



def bm25_search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Повертає [(arxiv_id, bm25_score)], відсортовані за спаданням."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(-scores)[:top_k]
    return [(str(df.iloc[i]["id"]), float(scores[i])) for i in top_idx]


def vector_search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Повертає [(arxiv_id, cosine_score)] з Pinecone."""
    q_vec = model.encode(
        [query],
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )[0].tolist()
    result = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    return [(m["metadata"]["arxiv_id"], m["score"]) for m in result["matches"]]


def rrf_fusion(ranked_lists: list[list[tuple[str, float]]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: об'єднує довільну кількість ранжованих списків.
    Формула: RRF(d) = Σ 1 / (k + rank(d))  для кожного списку, де d присутній."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Гібридний пошук: об'єднує BM25 і векторний через RRF."""
    return rrf_fusion([bm25_search(query, top_k), vector_search(query, top_k)])


def print_results(results: list[tuple[str, float]], label: str, score_label: str = "Скор"):
    """Виводить топ-DISPLAY_K результатів із назвою, роком, категорією і скором."""
    print(f"\n  [{label}]")
    for rank, (arxiv_id, score) in enumerate(results[:DISPLAY_K], 1):
        row      = id_to_row.get(arxiv_id)
        title    = str(row["title"])[:72]   if row is not None else "— не знайдено —"
        year     = int(row["year"])         if row is not None else "?"
        category = str(row["category"])     if row is not None else "?"
        print(f"  {rank}. [{year} | {category}] {title}")
        print(f"     {score_label}: {score:.5f}")


def run_comparison(query: str):
    """Запускає BM25, векторний і гібридний пошук для одного запиту та виводить порівняння."""
    print(f"\n{'=' * 80}")
    print(f"Запит: \"{query}\"")
    print(f"{'=' * 80}")
    print_results(bm25_search(query),   label="BM25",            score_label="BM25-скор")
    print_results(vector_search(query), label="Векторний пошук", score_label="Cosine")
    print_results(hybrid_search(query), label="Гібридний (RRF)", score_label="RRF-скор")


if __name__ == "__main__":
    queries = [
        "BERT fine-tuning",                                      # точний термін
        "Yann LeCun convolutional networks",                     # ім'я автора
        "making computers understand human emotions from text",  # перефразування
    ]
    for q in queries:
        run_comparison(q)
