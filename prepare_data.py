"""Ham destek verisini temizler ve sıfırdan ML eğitimi için hazırlar."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import canonicalize_category, save_records


INPUT_FILE = Path("/Users/rumeysa/Downloads/mobile_user_feedback_export.jsonl")
TRAIN_FILE = Path("train.jsonl")
TEST_FILE = Path("test.jsonl")
TEST_RATIO = 0.20
RANDOM_STATE = 42

COUNTRY_TAGS = {
    "Russia", "Turkey", "Brazil", "Colombia", "Mexico", "Italy",
    "Germany", "Poland", "Portugal", "Spain", "Ecuador", "Bolivia",
    "Chile", "Argentina", "India", "France", "Ukraine",
    "Pay from another country", "Huawei",
}


def clean_category(tags: str) -> str:
    if not tags or pd.isna(tags):
        return "General"
    tags = [tag.strip() for tag in str(tags).split(",")]
    cleaned = [tag for tag in tags if tag and tag not in COUNTRY_TAGS]
    # Bir kayıtta birden çok tag olabilir. Sınıflandırma için ana konuyu, yani
    # ülke etiketi çıkarıldıktan sonraki ilk tag'i hedef etiket kabul ediyoruz.
    return canonicalize_category(cleaned[0] if cleaned else "General")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Ham veri dosyası bulunamadı: {INPUT_FILE}")

    print("📂 Ham veri okunuyor...")
    raw_data = []
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                raw_data.append(json.loads(line))
    df = pd.DataFrame(raw_data)
    original_count = len(df)
    print(f"   Toplam kayıt: {original_count}")

    required_columns = {"content", "answer", "tags"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Ham veride eksik sütunlar: {', '.join(sorted(missing))}")

    df = df[~df["tags"].fillna("").str.contains(r"\bSpam\b", case=False, regex=True)]
    df = df[~df["tags"].fillna("").str.contains(r"\bDuplicated\b", case=False, regex=True)]
    df = df[df["content"].notna() & df["answer"].notna()]
    df = df[df["content"].astype(str).str.strip().ne("")]
    df = df[df["answer"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=["content", "answer"], keep="first")
    df["category"] = df["tags"].apply(clean_category)
    # Dışa aktarımdaki tarih sütunu sürüme göre farklı ad taşıyabilir. Varsa
    # en güncel destek yanıtını tercih edebilmek için aynen saklanır.
    date_column = next((column for column in (
        "updated_at", "updatedAt", "answered_at", "answeredAt", "created_at",
        "createdAt", "timestamp", "date",
    ) if column in df.columns), None)
    print(f"   Temiz kayıt: {len(df)} (silinen: {original_count - len(df)})")

    records = []
    for _, row in df.iterrows():
        record = {
            "text": str(row["content"]).strip(),
            "category": row["category"],
            "answer": str(row["answer"]).strip(),
        }
        if date_column and pd.notna(row[date_column]):
            record["updated_at"] = str(row[date_column]).strip()
        records.append(record)

    # Tek örnekli kategoriler testte olursa model o etiketi eğitimde hiç görmez.
    counts = Counter(record["category"] for record in records)
    train_only = [record for record in records if counts[record["category"]] == 1]
    splittable = [record for record in records if counts[record["category"]] > 1]
    labels = [record["category"] for record in splittable]

    try:
        train_part, test = train_test_split(
            splittable, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=labels
        )
    except ValueError:
        # Çok küçük bir veri kümesinde stratified split mümkün olmayabilir.
        train_part, test = train_test_split(
            splittable, test_size=TEST_RATIO, random_state=RANDOM_STATE
        )
    train = train_only + train_part

    save_records(train, TRAIN_FILE)
    save_records(test, TEST_FILE)
    print(f"✅ {TRAIN_FILE}: {len(train)} kayıt")
    print(f"✅ {TEST_FILE}: {len(test)} kayıt")
    print(f"   Kategori sayısı: {len(counts)}")
    print(f"   Tarih alanı: {date_column or 'yok (güncellik önceliği uygulanmaz)'}")
    print("\nYeni kayıt biçimi: {")
    print('  "text": "kullanıcı mesajı",')
    print('  "category": "etiket",')
    print('  "answer": "doğrulanmış destek yanıtı",')
    print('  "updated_at": "2026-08-17T14:30:00Z"  // varsa')
    print("}")


if __name__ == "__main__":
    main()
