"""Modeli arayüzden bağımsız, tekrarlanabilir güvenlik ve kalite kontrolleriyle denetler."""

from __future__ import annotations

from sklearn.metrics import accuracy_score

from chatbot_model import (
    EMAIL_PATTERN, LOW_QUALITY_REPLY_PATTERN, answer_message, clean_support_answer,
    is_safe_english_reply, load_chatbot, rank_with_recency,
)
import numpy as np
from data_utils import group_category, load_records


TEST_FILE = "test_model_data.jsonl"
# Kural tabanlı kısa mesajlar veri kümesindeki herhangi bir destek etiketine
# zorlanmamalıdır. Bu liste regresyon testidir: değişiklikte tekrar çalıştırılır.
BEHAVIOUR_CASES = {
    "merhaba": ("Welcome", "rule"),
    "nasılsın": ("Social conversation", "rule"),
    "nasilsin": ("Social conversation", "rule"),
    "thank you": ("Thanks", "rule"),
    "aa": ("Needs clarification", "none"),
    "???": ("Needs clarification", "none"),
    # Kısa yazım hataları ve anlamsız metinler hiçbir destek grubuna zorlanmaz.
    "mergabab": ("Needs clarification", "none"),
    "asdasd": ("Needs clarification", "none"),
    "şifremi nasıl değiştiririm?": ("Account, sign-in & email", "retrieval"),
    "How can I pay with PIX?": ("Payments & methods", "retrieval"),
    "rotamı paylaşmak istiyorum": ("Sharing & collaboration", "retrieval"),
}


def assert_pix_reply_quality(model: dict, failures: list[str]) -> None:
    """PIX ödeme sorusu için genel/bozuk ödeme yanıtı değil, doğru alt konuyu denetler."""
    result = answer_message(model, "pix ile nasıl öderim")
    best = result["suggestions"][0]
    if best.get("matched_specific_category") != "How to pay with PIX":
        failures.append("PIX ödeme sorusunda 'How to pay with PIX' yanıtı ilk öneri değil.")
    if LOW_QUALITY_REPLY_PATTERN.search(best["answer"]):
        failures.append("PIX ödeme sorusunda düşük kaliteli genel yanıt gösterildi.")


def assert_recency_ranking(failures: list[str]) -> None:
    """Güncel kayıt, yalnız benzer adaylar arasında öne geçmelidir."""
    similarities = np.array([0.78, 0.75, 0.50])
    ranked = rank_with_recency(
        similarities, similarities.copy(), np.array([0.0, 1.0, 1.0]),
    )
    if int(np.argmax(ranked)) != 1:
        failures.append("Yakın eşleşmelerde en güncel kayıt önceliklenmedi.")
    if ranked[2] != similarities[2]:
        failures.append("Alakasız düşük benzerlikli kayıt güncellik nedeniyle yükseltildi.")


def main() -> None:
    model = load_chatbot()
    test_records = load_records(TEST_FILE)
    if not test_records:
        raise ValueError("Test verisi bulunamadı.")

    failures: list[str] = []
    trained_groups = set(model["training_categories"])
    expected_groups = [group_category(record["category"]) for record in test_records]
    predicted_groups = model["classifier"].predict([record["text"] for record in test_records])
    unexpected = sorted(set(predicted_groups) - trained_groups)
    if unexpected:
        failures.append(f"Model eğitimde olmayan etiket döndürdü: {unexpected}")
    accuracy = accuracy_score(expected_groups, predicted_groups)

    # Bütün bellekteki yanıtlar ekrana çıkmadan önce e-posta kalıntısı bırakıyor mu?
    unsafe_answers = [
        index for index, answer in enumerate(model["training_answers"])
        if EMAIL_PATTERN.search(clean_support_answer(answer))
    ]
    if unsafe_answers:
        failures.append(f"{len(unsafe_answers)} öneride e-posta kalıntısı var.")
    non_english_answers = [
        index for index, answer in enumerate(model["training_answers"])
        if not is_safe_english_reply(answer)
    ]
    if non_english_answers:
        failures.append(f"{len(non_english_answers)} öneri İngilizce güvenlik denetimini geçemedi.")

    for message, (expected_category, expected_type) in BEHAVIOUR_CASES.items():
        result = answer_message(model, message)
        if (result["category"], result["match_type"]) != (expected_category, expected_type):
            failures.append(
                f"'{message}' -> ({result['category']}, {result['match_type']}); "
                f"beklenen ({expected_category}, {expected_type})"
            )
    assert_pix_reply_quality(model, failures)
    assert_recency_ranking(failures)

    # Farklı destek gruplarındaki cevapların birbiriyle karışmadığını örnek test
    # kümesinden kontrol ediyoruz. Bu, öneri filtrelemesinin regresyon kontrolüdür.
    sampled_records = test_records[::max(1, len(test_records) // 100)]
    for record in sampled_records:
        result = answer_message(model, record["text"])
        if result["match_type"] != "retrieval":
            continue
        if any(item["matched_category"] != result["category"] for item in result["suggestions"]):
            failures.append(f"Kategori dışı öneri üretildi: {record['text'][:60]}")
            break
        if any(EMAIL_PATTERN.search(item["answer"]) for item in result["suggestions"]):
            failures.append(f"Öneride e-posta üretildi: {record['text'][:60]}")
            break
        if any(not is_safe_english_reply(item["answer"]) for item in result["suggestions"]):
            failures.append(f"İngilizce dışı öneri üretildi: {record['text'][:60]}")
            break

    print("\nMODEL KALİTE KONTROLÜ")
    print(f"  Test örneği: {len(test_records)}")
    print(f"  Ana grup sınıflandırma doğruluğu: %{accuracy * 100:.1f}")
    print(f"  Eğitimdeki ana grup sayısı: {len(trained_groups)}")
    print(f"  E-posta güvenlik kontrolü: {'başarılı' if not unsafe_answers else 'başarısız'}")
    print(f"  İngilizce yanıt kontrolü: {'başarılı' if not non_english_answers else 'başarısız'}")
    print(f"  Kural / anlamsız mesaj kontrolleri: {'başarılı' if not failures else 'başarısız'}")
    if failures:
        print("\nHATALAR:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\n✅ Tüm zorunlu kalite kontrolleri geçti.")


if __name__ == "__main__":
    main()
