"""Çok dilli destek kayıtlarını toplu biçimde İngilizce hafızaya çevirir."""

from collections import defaultdict
from pathlib import Path

from data_utils import load_records, save_records
from translation_service import MEMORY_LANGUAGE, detect_language, translate_many


SOURCES = {
    Path("train.jsonl"): Path("train_english.jsonl"),
    Path("test.jsonl"): Path("test_english.jsonl"),
}


def translate_column(records: list[dict], field: str) -> list[str]:
    """Bir alanı kaynak dile göre gruplayıp toplu çevirir."""
    languages = [detect_language(record[field]) for record in records]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(languages):
        groups[language].append(index)

    output = [""] * len(records)
    for language, indices in sorted(groups.items()):
        print(f"   {field}: {language} ({len(indices)} kayıt)")
        for start in range(0, len(indices), 100):
            batch_indices = indices[start:start + 100]
            values = [records[index][field] for index in batch_indices]
            translations = translate_many(values, language, MEMORY_LANGUAGE)
            for index, translated in zip(batch_indices, translations):
                output[index] = translated
            print(f"      {language}: {min(start + len(batch_indices), len(indices))}/{len(indices)}")
    return output


def translate_records(records: list[dict], name: str) -> list[dict]:
    print(f"   {name}: dil algılama ve soru çevirisi...")
    english_texts = translate_column(records, "text")
    print(f"   {name}: yanıt çevirisi...")
    english_answers = translate_column(records, "answer")
    return [
        {
            "text": text,
            "category": record["category"],
            "answer": answer,
            "source_language": detect_language(record["text"]),
            **({"updated_at": record["updated_at"]} if record.get("updated_at") else {}),
        }
        for record, text, answer in zip(records, english_texts, english_answers)
    ]


def main() -> None:
    for source, destination in SOURCES.items():
        records = load_records(source)
        print(f"🌐 {source} → İngilizce hafıza hazırlanıyor ({len(records)} kayıt)...")
        save_records(translate_records(records, source.name), destination)
        print(f"✅ {destination} kaydedildi.")


if __name__ == "__main__":
    main()
