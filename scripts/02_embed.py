import pandas as pd
import numpy as np
import os

import torch



INPUT_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64
VECTOR_DIM = 768  # Розмірність векторів для allenai/specter2_base


def load_model():
    from sentence_transformers import SentenceTransformer
    model_name = MODEL_NAME
    return SentenceTransformer(model_name)


def prepare_texts(df: pd.DataFrame) -> list[str]:
    titles = df['title'].fillna('').astype(str)
    abstracts = df['abstract'].fillna('').astype(str)
    return [f"{title} [SEP] {abstract}" for title, abstract in zip(titles, abstracts)]


def embed_texts(texts, batch_size=64):
    model = load_model()

    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    # Перевірка наявності підготовленого файлу
    if not os.path.exists(INPUT_FILE):
        print(f"Помилка: файл {INPUT_FILE} не знайдено. Запустіть спочатку 01_prepare_data.py")
        exit(1)

    df = pd.read_parquet(INPUT_FILE)
    texts = prepare_texts(df)

    embeddings = embed_texts(texts, batch_size=BATCH_SIZE).astype(np.float32)

    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    np.save(EMBEDDINGS_FILE, embeddings)

    first_norm = float(np.linalg.norm(embeddings[0]))
    print(f"Норма першого вектора: {first_norm}")
    print(f"Розмірність векторів: {embeddings.shape[1]}")
    print(f"Збережено вектори у {EMBEDDINGS_FILE} з формою {embeddings.shape}")

