# scripts/01_prepare_data.py
import json
import os
import pandas as pd
from tqdm import tqdm

INPUT_FILE   = "data/arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE  = "data/arxiv_subset.parquet"
MAX_PER_YEAR = 280  # рівномірно по роках

os.makedirs("data", exist_ok=True)

def extract_year(paper: dict) -> int:
    """
    Беремо рік із першої версії статті — це дата публікації на arXiv.
    update_date — дата останнього оновлення, вона може бути на роки пізніше.
    Формат created: "Mon, 2 Apr 2007 19:18:42 GMT"
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]  # "Mon, 2 Apr 2007 19:18:42 GMT"
            # Рік стоїть на 4-й позиції після split по пробілу
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    # Запасний варіант: update_date у форматі "YYYY-MM-DD"
    return int(paper.get("update_date", "2000-01-01")[:4])

def format_authors(paper: dict) -> str:
    """
    authors_parsed — структурований список [["Прізвище", "Ініціали", ""]].
    Збираємо у читабельний рядок "Прізвище І., Прізвище І."
    Якщо authors_parsed відсутній — беремо сирий рядок authors.
    """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:  # не більше 10 авторів
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    # Запасний варіант: сирий рядок авторів
    return paper.get("authors", "").replace("\\n", " ")

records_by_year = {}
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаємо датасет"):
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = paper.get("abstract", "").strip()
        title    = paper.get("title", "").strip()

        if not abstract or not title:
            continue

        year = extract_year(paper)
        bucket = records_by_year.setdefault(year, [])
        if len(bucket) >= MAX_PER_YEAR:
            continue

        categories_raw = paper.get("categories", "unknown")
        primary_category = categories_raw.split()[0]

        bucket.append({
            "id":       paper["id"],
            "title":    title.replace("\\n", " ").strip(),
            "abstract": abstract.replace("\\n", " ").strip(),
            "authors":  format_authors(paper),
            "year":     year,
            "category": primary_category,
        })

records = [r for bucket in records_by_year.values() for r in bucket]

df = pd.DataFrame(records)
df.to_parquet(OUTPUT_FILE, index=False)

print(f"Завантажено статей: {len(df)}")
print(f"Роки: від {df['year'].min()} до {df['year'].max()}")
print(f"Збережено у {OUTPUT_FILE}")
print("\nРозподіл за роками (всі):")
print(df["year"].value_counts().sort_index().to_string())
print("\nРозподіл за категоріями (топ-10):")
print(df["category"].value_counts().head(10).to_string())
