"""Sıfırdan eğitilen müşteri destek modelini test kümesinde değerlendirir."""

import json
from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report

from chatbot_model import answer_message, load_chatbot
from data_utils import group_category, load_records


TEST_FILE = "test_model_data.jsonl"
RESULTS_FILE = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"


def main() -> None:
    model = load_chatbot()
    test_records = load_records(TEST_FILE)
    if not test_records:
        raise ValueError("Test için test.jsonl içinde geçerli kayıt bulunamadı.")

    # Model ince taneli (867 farklı) ham etiketi değil, iş açısından anlaşılır
    # ana grubu öğrenir. Test beklentisini de aynı şemaya getiriyoruz.
    test_records = [
        {**record, "category": group_category(record["category"])}
        for record in test_records
    ]
    print(f"📊 {len(test_records)} test örneği değerlendiriliyor...")
    results = []
    expected_categories = []
    predicted_categories = []

    # Sınıflandırma doğruluğu, tek tek HTTP/arayüz çağrısı değil modelin doğrudan
    # testidir. Yanıt getirmenin kalitesi aşağıda örneklenmiş kayıtlarla ayrıca izlenir.
    classifier_groups = model["classifier"].predict([record["text"] for record in test_records])
    # Kapsamlı sınıflandırma metriği vektör hâlinde hızlı çalışır. Yanıt bulma
    # işlemi, maliyeti sınırlı ama temsilî 100 kayıtta ayrı denetlenir.
    for index, (record, predicted_group) in enumerate(zip(test_records, classifier_groups), start=1):
        expected_categories.append(record["category"])
        predicted_categories.append(str(predicted_group))
    retrieval_records = test_records[::max(1, len(test_records) // 100)]
    for index, record in enumerate(retrieval_records, start=1):
        prediction = answer_message(model, record["text"])
        results.append({
            "id": index,
            "user_message": record["text"],
            "expected_category": record["category"],
            "predicted_category": prediction["category"],
            "category_correct": prediction["category"] == record["category"],
            "expected_answer": record["answer"],
            "retrieved_answer": prediction["answer"],
            "matched_question": prediction["matched_question"],
            "similarity": round(prediction["similarity"], 4),
        })

    accuracy = accuracy_score(expected_categories, predicted_categories)
    report = classification_report(
        expected_categories, predicted_categories, zero_division=0, output_dict=True
    )
    summary = {
        "test_examples": len(test_records),
        "category_accuracy": round(float(accuracy), 4),
        "mean_retrieval_similarity": round(
            sum(result["similarity"] for result in results) / len(results), 4
        ),
        "model": model["metadata"],
    }

    print("\n" + "=" * 54)
    print("📊 DEĞERLENDİRME SONUÇLARI")
    print("=" * 54)
    print(f"  Test örneği: {summary['test_examples']}")
    print(f"  Kategori doğruluğu: %{accuracy * 100:.1f}")
    print(f"  Yanıt bulma kontrol örneği: {len(results)} kayıt")
    print(f"  Ortalama eşleşme benzerliği: %{summary['mean_retrieval_similarity'] * 100:.1f}")
    print("=" * 54)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "classification_report": report, "results": results},
            file, ensure_ascii=False, indent=2,
        )
    print(f"💾 Ayrıntılı sonuçlar kaydedildi: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
