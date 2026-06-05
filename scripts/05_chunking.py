# scripts/05_chunking.py
import os
import re
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME     = "allenai/specter2_base"
VECTOR_DIM     = 768
INDEX_FIXED    = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
TOP_N_PAPERS   = 30
CHUNK_SIZE     = 100   # слів у fixed-чанку
OVERLAP        = 20    # слів перекриття між fixed-чанками
MAX_WORDS      = 150   # максимум слів у semantic-чанку
BATCH_SIZE     = 100
TOP_K          = 5

pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df    = pd.read_parquet("data/arxiv_subset.parquet")


def fixed_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Розбиває текст на чанки фіксованого розміру (у словах) з перекриттям."""
    words = text.split()
    step  = chunk_size - overlap
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += step
    return chunks


def semantic_chunks(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Об'єднує повні речення у чанки, не перевищуючи max_words слів."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def get_or_create_index(name: str, dimension: int, metric: str = "cosine"):
    """Повертає наявний або створює новий індекс Pinecone."""
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        print(f"Створюю індекс '{name}'...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status.get("ready", False):
            time.sleep(1)
    else:
        print(f"Індекс '{name}' вже існує.")
    return pc.Index(name)


def upsert_chunks(index, records: list[dict]):
    """Обчислює ембеддинги для всіх чанків і завантажує їх у Pinecone батчами."""
    texts = [r["chunk_text"] for r in records]
    print(f"  Обчислюємо ембеддинги для {len(texts)} чанків...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    vectors = [
        {
            "id": r["chunk_id"],
            "values": emb.tolist(),
            "metadata": {k: v for k, v in r.items() if k != "chunk_id"},
        }
        for r, emb in zip(records, embeddings)
    ]
    for start in tqdm(range(0, len(vectors), BATCH_SIZE), desc="  Завантаження батчів"):
        index.upsert(vectors=vectors[start: start + BATCH_SIZE])

def search_chunks(query: str, index, label: str):
    """Виконує семантичний пошук по індексу чанків і виводить топ-5 результатів."""
    print(f"\n[{label}] Запит: '{query}'")
    q_vec = model.encode(
        [query],
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )[0].tolist()
    result = index.query(vector=q_vec, top_k=TOP_K, include_metadata=True)
    for rank, match in enumerate(result["matches"], 1):
        meta    = match["metadata"]
        snippet = str(meta.get("chunk_text", ""))[:200]
        print(f"  {rank}. {meta.get('title', '—')} (чанк {int(meta.get('chunk_num', 0))})")
        print(f"     {snippet}...")
        print(f"     Схожість: {match['score']:.4f}")

if __name__ == "__main__":
    # 1. Вибираємо 30 статей із найдовшими анотаціями
    top_papers = (
        df.assign(_len=df["abstract"].str.len())
          .nlargest(TOP_N_PAPERS, "_len")
          .drop(columns="_len")
          .reset_index(drop=True)
    )
    abs_lens = top_papers["abstract"].str.len()
    print(f"Обрано {len(top_papers)} статей. Довжина анотацій: {abs_lens.min()}–{abs_lens.max()} символів")

    # 2. Чанкінг двома стратегіями
    fixed_records, semantic_records = [], []
    for _, row in top_papers.iterrows():
        text = str(row["abstract"])
        year = int(row["year"]) if pd.notna(row["year"]) else 0
        base = {
            "arxiv_id": str(row["id"]),
            "title":    str(row["title"]),
            "year":     year,
            "category": str(row["category"]),
        }
        for i, chunk in enumerate(fixed_chunks(text)):
            fixed_records.append({
                "chunk_id":   f"{row['id']}_fixed_{i}",
                "chunk_text": chunk,
                "chunk_num":  i,
                **base,
            })
        for i, chunk in enumerate(semantic_chunks(text)):
            semantic_records.append({
                "chunk_id":   f"{row['id']}_semantic_{i}",
                "chunk_text": chunk,
                "chunk_num":  i,
                **base,
            })

    print(f"Fixed чанків: {len(fixed_records)} | Semantic чанків: {len(semantic_records)}")

    # 3. Створюємо або відкриваємо індекси
    idx_fixed    = get_or_create_index(INDEX_FIXED,    VECTOR_DIM)
    idx_semantic = get_or_create_index(INDEX_SEMANTIC, VECTOR_DIM)

    # 4 + 5. Ембеддинги та завантаження в Pinecone
    print("\nЗавантажуємо fixed чанки...")
    upsert_chunks(idx_fixed, fixed_records)

    print("\nЗавантажуємо semantic чанки...")
    upsert_chunks(idx_semantic, semantic_records)

    # 6. Тестові запити для обох типів індексів
    test_queries = [
        "neural networks for image classification",
        "natural language processing with transformers",
        "reinforcement learning in robotics",
    ]
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТИ ПОШУКУ")
    print("=" * 80)
    for query in test_queries:
        search_chunks(query, idx_fixed,    INDEX_FIXED)
        search_chunks(query, idx_semantic, INDEX_SEMANTIC)
        print("-" * 80)
