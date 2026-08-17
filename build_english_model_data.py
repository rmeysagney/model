"""Çok dilli soruları İngilizce yanıtlarla eşleştirerek model verisini kurar."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from data_utils import load_records, save_records


REFERENCE_FILE = Path("english_answer_reference.jsonl")
SOURCE_FILES = (Path("train.jsonl"), Path("test.jsonl"))
MODEL_FILE = Path("customer_support_ml/model.joblib")
TRAIN_FILE = Path("train_model_data.jsonl")
TEST_FILE = Path("test_model_data.jsonl")
TEST_RATIO = 0.20
RANDOM_STATE = 42


def record_key(record: dict) -> tuple[str, str]:
    return record["text"].strip(), record["answer"].strip()


def create_reference() -> None:
    """Hizalı İngilizce yanıt verisini, gerekirse mevcut modeli referans alır."""
    source_records = load_records("train.jsonl")
    english_records = load_records("train_english.jsonl")
    if len(source_records) == len(english_records):
        english_answers = [record["answer"] for record in english_records]
        source_name = "train_english.jsonl"
    elif MODEL_FILE.exists():
        artifact = joblib.load(MODEL_FILE)
        english_answers = artifact["training_answers"]
        source_name = "model.joblib"
    else:
        raise ValueError("İngilizce yanıt referansı için hizalı veri veya model bulunamadı.")
    if len(source_records) != len(english_answers):
        raise ValueError("İngilizce yanıt referansının satır sayısı train.jsonl ile eşleşmiyor.")
    lookup = {
        record_key(source): english_answer
        for source, english_answer in zip(source_records, english_answers)
    }

    with REFERENCE_FILE.open("w", encoding="utf-8") as file:
        for (text, answer), english_answer in lookup.items():
            file.write(json.dumps({
                "source_text": text,
                "source_answer": answer,
                "english_answer": english_answer,
            }, ensure_ascii=False) + "\n")
    print(f"✅ İngilizce yanıt referansı kaydedildi: {REFERENCE_FILE} ({len(lookup)} kayıt, kaynak: {source_name})")


def load_reference() -> dict[tuple[str, str], str]:
    lookup = {}
    with REFERENCE_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                lookup[(item["source_text"], item["source_answer"])] = item["english_answer"]
    return lookup


def build_records(lookup: dict[tuple[str, str], str]) -> list[dict]:
    output = []
    missing = 0
    for source_path in SOURCE_FILES:
        records = load_records(source_path)
        for record in records:
            english_answer = lookup.get(record_key(record))
            if not english_answer:
                missing += 1
                continue
            # Soru özgün dilinde kalır; model çok dilli girdiyi doğrudan öğrenir.
            # Yanıt ise tüm örneklerde İngilizcedir.
            output.append({
                "text": record["text"],
                "category": record["category"],
                "answer": english_answer,
            })
    if missing:
        print(f"⚠️ İngilizce yanıtı olmayan {missing} eski test kaydı dışarıda bırakıldı.")
    return output


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    counts = Counter(record["category"] for record in records)
    always_train = [record for record in records if counts[record["category"]] == 1]
    splittable = [record for record in records if counts[record["category"]] > 1]
    labels = [record["category"] for record in splittable]
    train_part, test = train_test_split(
        splittable, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=labels
    )
    return always_train + train_part, test


def save_model_splits(lookup: dict[tuple[str, str], str]) -> None:
    records = build_records(lookup)
    train, test = split_records(records)
    save_records(train, TRAIN_FILE)
    save_records(test, TEST_FILE)
    print(f"✅ {TRAIN_FILE}: {len(train)} kayıt (%80 eğitim)")
    print(f"✅ {TEST_FILE}: {len(test)} kayıt (%20 test)")


def main() -> None:
    if not REFERENCE_FILE.exists():
        create_reference()
    lookup = load_reference()
    save_model_splits(lookup)


if __name__ == "__main__":
    main()
