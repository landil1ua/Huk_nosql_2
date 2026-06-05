import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів
ABSTRACT_LIMIT = 500
AUTHORS_LIMIT = 200




def create_load_index(pc, name, dimension, metric):
    """Створює індекс у Pinecone, якщо він ще не існує, або повертає наявний."""
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        print(f"[common] Індекса '{name}' немає, створюю (dim={dimension}, {metric})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Створення не миттєве — чекаємо, поки індекс справді підніметься.
        import time
        while not pc.describe_index(name).status.get("ready", False):
            time.sleep(1)
    else:
        print(f"[common] Індекс '{name}' вже є, використовую.")
    return pc.Index(name)


def build_metadata(row):
    """Формує словник метаданих для вектора: використовується для фільтрації (year, category) та відображення результатів (title, abstract)."""
    year = row["year"]
    return {
        "arxiv_id": str(row["id"]),
        "title": str(row["title"]),
        "abstract": str(row["abstract"])[:ABSTRACT_LIMIT],
        "authors": str(row["authors"])[:AUTHORS_LIMIT],
        "year": int(year) if pd.notna(year) else 0,  # NaN -> 0, інакше впаде
        "category": str(row["category"]),
    }


if __name__ == "__main__":
    # Перевірка наявності підготовленого файлу
    if not os.path.exists(INPUT_PARQUET):
        print(f"Помилка: файл {INPUT_PARQUET} не знайдено. Запустіть спочатку 01_prepare_data.py")
        exit(1)

    if not os.path.exists(INPUT_EMBEDDINGS):
        print(f"Помилка: файл {INPUT_EMBEDDINGS} не знайдено. Запустіть спочатку 02_embed.py")
        exit(1)

    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    # Ініціалізація клієнта
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = INDEX_NAME
    dimension = embeddings.shape[1]
    metric = "cosine"

    index = create_load_index(pc, index_name, dimension, metric)

    vectors_to_upsert = []

    for i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Створюємо вектори для завантаження"):
        vectors_to_upsert.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": build_metadata(row)
        })
    # Завантажуємо всі вектори одним запитом
    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE), desc=f"Завантажуємо батчами по {BATCH_SIZE}"):
        batch = vectors_to_upsert[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)

    stats = index.describe_index_stats()
    print(f"Загальна кількість векторів в індексі: {stats['total_vector_count']}")