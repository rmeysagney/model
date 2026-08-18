"""önceden eğitilmiş model kullanmadan müşteri destek modelini eğitir. önce sayılara dönüşmesi gerekir. TfidfVectorizer her mesajdan karakter parçaları çıkarıyor.
“aboneliğimi iptal etmek istiyorum”
abo, abon, abone, abonei, ipt, ipta, iptal...
"""

from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from chatbot_model import clean_source_text, clean_support_answer, is_safe_english_reply
from data_utils import canonicalize_category, group_record_category, load_records, normalise_support_text


TRAIN_FILE = Path("train_model_data.jsonl")
MODEL_FILE = Path("customer_support_ml/model_english_multilingual.joblib")
DEPLOY_MODEL_FILE = Path("deployment_assets/model_english_multilingual.joblib")
VALIDATION_RATIO = 0.12
RANDOM_STATE = 42
SELECTION_TOLERANCE = 0.003
RETRIEVAL_META_CATEGORY_PREFIXES = ("replied via", "reply sent", "phone support", "contact via")

# Karakter ve kelime özelliklerini; ayrıca iki farklı doğrusal sınıflandırıcıyı
# aynı doğrulamada karşılaştırıyoruz. Seçim ada göre değil test sonucuna göre.
CANDIDATES = [
    {"name": "karakter_svc", "features": "char", "algorithm": "linear_svc", "ngram_range": (2, 6), "min_df": 2, "C": 1.0},
    {"name": "hibrit_svc", "features": "hybrid", "algorithm": "linear_svc", "ngram_range": (2, 6), "min_df": 2, "C": 1.0},
    {"name": "hibrit_genis_svc", "features": "hybrid", "algorithm": "linear_svc", "ngram_range": (3, 6), "min_df": 2, "C": 1.5},
    {"name": "hibrit_logistik", "features": "hybrid", "algorithm": "sgd_log", "ngram_range": (2, 6), "min_df": 2, "alpha": 0.00001},
]


def build_classifier(settings: dict) -> Pipeline:
    char_features = TfidfVectorizer(
        analyzer="char_wb", ngram_range=settings["ngram_range"],
        min_df=settings["min_df"], max_features=120_000, sublinear_tf=True,
    )
    if settings["features"] == "hybrid":
        features = FeatureUnion([
            ("characters", char_features),
            ("words", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=settings["min_df"],
                max_features=80_000, sublinear_tf=True,
            )),
        ], transformer_weights={"characters": 0.7, "words": 1.3})
    else:
        features = char_features
    classifier = (
        LinearSVC(C=settings["C"], class_weight="balanced", max_iter=10_000)
        if settings["algorithm"] == "linear_svc"
        else SGDClassifier(
            loss="log_loss", alpha=settings["alpha"], class_weight="balanced",
            max_iter=3_000, tol=1e-4, random_state=RANDOM_STATE,
        )
    )
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def parse_record_timestamp(value: object) -> float | None:
    """Opsiyonel kayıt tarihini sıralamada kullanılabilecek UTC saniyesine çevirir.

    Tarih biçimi geçersizse veya kaynakta yoksa ``None`` döner. Böylece kayıt
    sırası ya da dosya değiştirilme zamanı sahte "güncellik" olarak kullanılmaz.
    """
    if value in (None, ""):
        return None
    raw = str(value).strip()
    try:
        if raw.replace(".", "", 1).isdigit():
            numeric = float(raw)
            # Milisaniye cinsinden Unix zaman damgalarını destekle.
            return numeric / 1000 if numeric > 10_000_000_000 else numeric
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def make_recency_scores(records: list[dict]) -> tuple[list[float], int]:
    """Gerçek tarihli kayıtları 0–1 güncellik puanına dönüştürür."""
    timestamps = [parse_record_timestamp(record.get("updated_at")) for record in records]
    known = [timestamp for timestamp in timestamps if timestamp is not None]
    if not known:
        return [0.0] * len(records), 0
    oldest, newest = min(known), max(known)
    if newest == oldest:
        return [1.0 if timestamp is not None else 0.0 for timestamp in timestamps], len(known)
    return [
        (timestamp - oldest) / (newest - oldest) if timestamp is not None else 0.0
        for timestamp in timestamps
    ], len(known)


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


def is_displayable_retrieval_record(record: dict) -> bool:
    """Kanal/metin kalıntısı taşıyan kayıtları öneri belleğine alma."""
    specific = str(record.get("specific_category", "")).casefold().strip()
    return (
        is_safe_english_reply(record["answer"])
        and not specific.startswith(RETRIEVAL_META_CATEGORY_PREFIXES)
    )


def main() -> None:
    records = [
        {
            **record,
            "text": clean_source_text(record["text"]),
            "specific_category": canonicalize_category(record["category"]),
            "category": group_record_category(record["category"], clean_source_text(record["text"])),
        }
        for record in load_records(TRAIN_FILE)
    ]
    if len(records) < 2:
        raise ValueError("Eğitim için train_model_data.jsonl içinde en az iki geçerli kayıt gerekir.")

    texts = [normalise_support_text(record["text"]) for record in records]
    categories = [record["category"] for record in records]

    print(f"📂 {len(records)} eğitim kaydı yüklendi.")
    selection_train, validation = split_for_validation(records)
    validation_texts = [normalise_support_text(record["text"]) for record in validation]
    validation_categories = [record["category"] for record in validation]
    print(f"🔁 {len(CANDIDATES)} aday model iç doğrulamada tekrar eğitiliyor...")

    candidate_scores = []
    for index, settings in enumerate(CANDIDATES, start=1):
        print(f"   [{index}/{len(CANDIDATES)}] {settings['name']} deneniyor...")
        candidate = build_classifier(settings)
        candidate.fit(
            [normalise_support_text(record["text"]) for record in selection_train],
            [record["category"] for record in selection_train],
        )
        score = accuracy_score(validation_categories, candidate.predict(validation_texts))
        candidate_scores.append({**settings, "validation_accuracy": round(float(score), 4)})
        print(f"       İç doğrulama doğruluğu: %{score * 100:.1f}")

    # Fark %0,3 puandan küçükse daha karmaşık modelin kazancı istatistiksel
    # olarak anlamlı kabul edilmez. Bu durumda daha yalın karakter-SVC modeli
    # seçilir; hem daha kararlı hem de bağımsız testte daha iyi geneller.
    best_validation = max(result["validation_accuracy"] for result in candidate_scores)
    close_candidates = [
        result for result in candidate_scores
        if result["validation_accuracy"] >= best_validation - SELECTION_TOLERANCE
    ]
    selected_settings = min(
        close_candidates,
        key=lambda result: (result["features"] != "char", result["algorithm"] != "linear_svc"),
    )
    print(f"🏆 Seçilen ayar: {selected_settings['name']}")
    print("🧠 Seçilen ayarla tüm eğitim verisinde nihai model eğitiliyor...")
    classifier = build_classifier(selected_settings)
    classifier.fit(texts, categories)

    print("🔎 Yanıt bulucu eğitiliyor...")
    response_vectorizer = FeatureUnion([
        ("characters", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            max_features=160_000, sublinear_tf=True,
        )),
        ("words", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            max_features=100_000, sublinear_tf=True,
        )),
    ], transformer_weights={"characters": 0.7, "words": 1.3})
    # Sınıflandırıcı tüm veriyle eğitilir. Yanıt belleğine yalnızca güvenli,
    # İngilizce gösterilebilecek kayıtlar alınır.
    retrieval_records = [record for record in records if is_displayable_retrieval_record(record)]
    retrieval_texts = [record["text"] for record in retrieval_records]
    retrieval_model_texts = [normalise_support_text(record["text"]) for record in retrieval_records]
    retrieval_categories = [record["category"] for record in retrieval_records]
    retrieval_specific_categories = [record["specific_category"] for record in retrieval_records]
    retrieval_answers = [clean_support_answer(record["answer"]) for record in retrieval_records]
    retrieval_dates = [str(record.get("updated_at", "")).strip() for record in retrieval_records]
    retrieval_recency, records_with_dates = make_recency_scores(retrieval_records)
    response_matrix = response_vectorizer.fit_transform(retrieval_model_texts)

    artifact = {
        "classifier": classifier,
        "response_vectorizer": response_vectorizer,
        "response_matrix": response_matrix,
        "training_texts": retrieval_texts,
        "training_categories": retrieval_categories,
        "training_specific_categories": retrieval_specific_categories,
        "training_answers": retrieval_answers,
        "training_updated_at": retrieval_dates,
        "training_recency_scores": retrieval_recency,
        "metadata": {
            "algorithm": "Domain-normalized TF-IDF words + character n-grams + selected linear classifier + cosine retrieval",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_examples": len(records),
            "retrieval_examples": len(retrieval_records),
            "category_count": len(set(categories)),
            "pretrained_model": False,
            "memory_language": "en",
            "input_languages": "multilingual",
            "response_language": "en",
            "label_scheme": "business-oriented grouped support labels",
            "text_representation": "Original multilingual text plus domain intent markers; no pretrained model or runtime translation.",
            "date_aware_ranking": {
                "records_with_dates": records_with_dates,
                "policy": "Within close semantic matches, prefer the most recently updated record.",
            },
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
