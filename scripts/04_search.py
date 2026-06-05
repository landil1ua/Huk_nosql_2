# scripts/04_search.py
from datetime import datetime
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch


load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
EMB_FILE = "embeddings/embeddings.npy"
PRQ_FILE = "data/arxiv_subset.parquet"
TOP_K = 5
RECENT_YEARS = 5


def print_results(results):
    """Виводить список результатів пошуку у зручному форматі."""
    if not results:
        print("Немає результатів для цього запиту.")
        return
    for rank, match in enumerate(results, 1):

        meta = match["metadata"]
        abstract = str(meta.get("abstract", ""))[:200]
        # Виводимо результати з назвою, категорією, роком і частиною абстракту.
        print(f"{meta.get('title', 'Без назви')}")
        print(f"рік: {meta.get('year', 'N/A')}, категорія: {meta.get('category', 'N/A')}")
        print(f"Абстракт: {abstract}...")
        print(f"Нормалізована схожість: {match['score']:.4f}")
        print("-" * 80)


def embed_texts(texts, model, batch_size):
    """Кодує список текстів у нормалізовані вектори за допомогою sentence-transformers."""
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def embed_query(query, model):
    """Повертає вектор одного запиту."""
    return embed_texts([query], model, batch_size=1)[0]


def semantic_search(query, index, model, top_k=5):
    """Виконує семантичний пошук у Pinecone за текстовим запитом."""
    print(f"Пошук за запитом: '{query}'")
    vector = embed_query(query, model)
    result = index.query(
        vector=vector.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    print_results(result["matches"])


def filtered_search(query, index, model, category=None, recent_years=None, to_year=None, top_k=5):
    """Пошук із фільтрацією за роком і, опціонально, категорією.
    category=None — будь-яка категорія (фільтр не застосовується)."""
    vector = embed_query(query, model).tolist()
    if recent_years is not None:
        cur_year = datetime.now().year
        threshold = cur_year - recent_years
        pinecone_filter = {"year": {"$gte": threshold}}
        if category:
            pinecone_filter["category"] = {"$eq": category}
        cat_label = f", категорія {category}" if category else ", будь-яка категорія"
        print(f"Пошук статей після {threshold} року{cat_label}...")
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
        print_results(result["matches"])
    if to_year is not None:
        pinecone_filter = {"year": {"$lte": to_year}}
        if category:
            pinecone_filter["category"] = {"$eq": category}
        cat_label = f", категорія {category}" if category else ", будь-яка категорія"
        print(f"Пошук статей до {to_year} року{cat_label}...")
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
        print_results(result["matches"])


def local_metric_comparison(query, model):
    """Порівнює три метрики подібності (cosine, dot product, L2) на локально збережених векторах.
    На нормалізованих векторах топи усіх трьох метрик мають збігатися."""
    print(f"\n=== Локально порівнюємо метрики для: \"{query}\" ===")
    embeddings = np.load(EMB_FILE)
    df = pd.read_parquet(PRQ_FILE)
    q = embed_query(query, model)

    dot = embeddings @ q                        # скалярний добуток
    norms = np.linalg.norm(embeddings, axis=1)
    cosine = dot / (norms * np.linalg.norm(q))  # ділимо на довжини векторів
    l2 = np.linalg.norm(embeddings - q, axis=1) # евклідова відстань

    # cosine та dot сортуємо за спаданням (більше = ближче), L2 за зростанням
    metrics = {
        "cosine (спад.)": np.argsort(-cosine),
        "dot product (спад.)": np.argsort(-dot),
        "L2 distance (зрост.)": np.argsort(l2),
    }
    for name, order in metrics.items():
        print(f"\n  -- {name} --")
        for rank, idx in enumerate(order[:TOP_K], 1):
            title = df.iloc[idx]["title"]
            print(f"    {rank}. [{idx}] {title[:80]}")



if __name__ == "__main__":
    # Підключення до Pinecone та завантаження індексу
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_parquet(PRQ_FILE)  # для отримання повного abstract

    # Приклад пошуку (чистий семантичний пошук: без фільтрів)
    user_query = "teaching machines to recognize objects in pictures"
    semantic_search(user_query, index, model, top_k=TOP_K)

    # Приклад A: статті по reinforcement learning за останні 5 років, категорія cs.LG
    rl_query = "reinforcement learning"
    print("\n=== Приклад A ===")
    filtered_search(rl_query, index, model, category="cs.LG", recent_years=5, top_k=TOP_K)

    # Приклад B: старі статті до 2015 року, будь-яка категорія
    print("\n=== Приклад B ===")
    filtered_search(rl_query, index, model, to_year=2015, top_k=TOP_K)

    # Порівняння метрик на локальних векторах
    print("\n=== Порівняння метрик ===")
    local_metric_comparison(user_query, model)



