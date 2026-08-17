"""önceden eğitilmiş model kullanmadan müşteri destek modelini eğitir. önce sayılara dönüşmesi gerekir. TfidfVectorizer her mesajdan karakter parçaları çıkarıyor.
“aboneliğimi iptal etmek istiyorum”
abo, abon, abone, abonei, ipt, ipta, iptal...
"""

from datetime import datetime
from pathlib import Path
from collections import Counter
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from chatbot_model import clean_source_text, clean_support_answer, is_safe_english_reply
from data_utils import canonicalize_category, group_category, load_records


TRAIN_FILE = Path("train_model_data.jsonl")
MODEL_FILE = Path("customer_support_ml/model_english_multilingual.joblib")
DEPLOY_MODEL_FILE = Path("deployment_assets/model_english_multilingual.joblib")
VALIDATION_RATIO = 0.12
RANDOM_STATE = 42

# Farklı n-gram genişliği ve C düzenleme gücüyle aday model denemeleri.
CANDIDATES = [
    {"name": "denge", "ngram_range": (3, 5), "min_df": 2, "C": 1.0},
    {"name": "kisa_n_gram", "ngram_range": (2, 5), "min_df": 2, "C": 0.8},
    {"name": "genis_n_gram", "ngram_range": (3, 6), "min_df": 2, "C": 1.5},
    {"name": "genis_kapsama", "ngram_range": (2, 6), "min_df": 2, "C": 1.0},
]


def build_classifier(settings: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=settings["ngram_range"],
            min_df=settings["min_df"], max_features=120_000, sublinear_tf=True,
        )),
        ("classifier", LinearSVC(
            C=settings["C"], class_weight="balanced", max_iter=10_000
        )),
    ])


def split_for_validation(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Tek örnekli sınıfları yalnızca eğitimde tutarak adil iç test oluşturur."""
    counts = Counter(record["category"] for record in records)
    always_train = [record for record in records if counts[record["category"]] == 1]
    splittable = [record for record in records if counts[record["category"]] > 1]
    labels = [record["category"] for record in splittable]
    train_part, validation = train_test_split(
        splittable, test_size=VALIDATION_RATIO, random_state=RANDOM_STATE, stratify=labels
    )
    return always_train + train_part, validation


def main() -> None:
    records = [
        {
            **record,
            "text": clean_source_text(record["text"]),
            "specific_category": canonicalize_category(record["category"]),
            "category": group_category(record["category"]),
        }
        for record in load_records(TRAIN_FILE)
    ]
    if len(records) < 2:
        raise ValueError("Eğitim için train_model_data.jsonl içinde en az iki geçerli kayıt gerekir.")

    texts = [record["text"] for record in records]
    categories = [record["category"] for record in records]

    print(f"📂 {len(records)} eğitim kaydı yüklendi.")
    selection_train, validation = split_for_validation(records)
    validation_texts = [record["text"] for record in validation]
    validation_categories = [record["category"] for record in validation]
    print(f"🔁 {len(CANDIDATES)} aday model iç doğrulamada tekrar eğitiliyor...")

    candidate_scores = []
    for index, settings in enumerate(CANDIDATES, start=1):
        print(f"   [{index}/{len(CANDIDATES)}] {settings['name']} deneniyor...")
        candidate = build_classifier(settings)
        candidate.fit(
            [record["text"] for record in selection_train],
            [record["category"] for record in selection_train],
        )
        score = accuracy_score(validation_categories, candidate.predict(validation_texts))
        candidate_scores.append({**settings, "validation_accuracy": round(float(score), 4)})
        print(f"       İç doğrulama doğruluğu: %{score * 100:.1f}")

    selected_settings = max(candidate_scores, key=lambda result: result["validation_accuracy"])
    print(f"🏆 Seçilen ayar: {selected_settings['name']}")
    print("🧠 Seçilen ayarla tüm eğitim verisinde nihai model eğitiliyor...")
    classifier = build_classifier(selected_settings)
    classifier.fit(texts, categories)

    print("🔎 Yanıt bulucu eğitiliyor...")
    response_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1,
        max_features=160_000, sublinear_tf=True,
    )
    # Sınıflandırıcı tüm veriyle eğitilir. Yanıt belleğine yalnızca güvenli,
    # İngilizce gösterilebilecek kayıtlar alınır.
    retrieval_records = [record for record in records if is_safe_english_reply(record["answer"])]
    retrieval_texts = [record["text"] for record in retrieval_records]
    retrieval_categories = [record["category"] for record in retrieval_records]
    retrieval_specific_categories = [record["specific_category"] for record in retrieval_records]
    retrieval_answers = [clean_support_answer(record["answer"]) for record in retrieval_records]
    response_matrix = response_vectorizer.fit_transform(retrieval_texts)

    artifact = {
        "classifier": classifier,
        "response_vectorizer": response_vectorizer,
        "response_matrix": response_matrix,
        "training_texts": retrieval_texts,
        "training_categories": retrieval_categories,
        "training_specific_categories": retrieval_specific_categories,
        "training_answers": retrieval_answers,
        "metadata": {
            "algorithm": "TF-IDF character n-grams + LinearSVC + cosine retrieval",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_examples": len(records),
            "retrieval_examples": len(retrieval_records),
            "category_count": len(set(categories)),
            "pretrained_model": False,
            "memory_language": "en",
            "input_languages": "multilingual",
            "response_language": "en",
            "label_scheme": "business-oriented grouped support labels",
            "model_selection": {
                "validation_ratio": VALIDATION_RATIO,
                "candidates": candidate_scores,
                "selected": selected_settings,
            },
        },
    }
    for output_file in (MODEL_FILE, DEPLOY_MODEL_FILE):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Çalışan arayüz model dosyasını aynı anda okuyabilir. Önce tam bir geçici
        # dosyaya yazıp sonra atomik olarak değiştiriyoruz; yarım pickle okunamaz.
        temporary_model = output_file.with_suffix(output_file.suffix + ".tmp")
        joblib.dump(artifact, temporary_model, compress=3)
        os.replace(temporary_model, output_file)

    print("\n✅ Eğitim tamamlandı.")
    print(f"   Kategori sayısı: {len(set(categories))}")
    print(f"   Kaydedilen model: {MODEL_FILE}")
    print(f"   Yayın modeli: {DEPLOY_MODEL_FILE}")
    print("   Not: Bu model yalnızca eğitim verisinden öğrenir; harici hazır model kullanmaz.")


if __name__ == "__main__":
    main()
