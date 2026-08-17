"""Eğitilmiş sınıflandırıcı ve yanıt bulucuyu kullanma fonksiyonları."""

from pathlib import Path
import re

import joblib
import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect_langs
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import group_category


DEFAULT_MODEL_FILE = Path("customer_support_ml/model_english_multilingual.joblib")
DetectorFactory.seed = 42
SUPPORTED_LANGUAGES = {
    "en", "tr", "pt", "es", "ru", "it", "fr", "de", "pl", "ro", "id", "ar", "uk", "nl"
}
GREETING_WORDS = {
    "merhaba", "merhabalar", "selam", "selamlar", "hello", "hi", "hey", "hola",
    "olá", "ola", "bonjour", "ciao", "hallo", "guten tag", "привет", "مرحبا",
}
SOCIAL_PHRASES = {
    "nasılsın", "nasilsin", "nasılsınız", "nasilsiniz", "naber", "ne haber", "ne yapıyorsun",
    "how are you", "how r you", "what s up", "whats up", "como estas", "como está",
}
THANKS_WORDS = {
    "teşekkürler", "teşekkür", "sağol", "sağ ol", "thanks", "thank you", "gracias",
    "merci", "grazie", "obrigado", "obrigada", "danke", "спасибо", "شكرا",
}
MIN_RETRIEVAL_SIMILARITY = 0.12
EMAIL_PATTERN = re.compile(
    r"(?ix)(?<![\w@])[\w.*%+\-]{1,128}\s*@\s*"
    r"(?:[a-z0-9\-]{1,63}\s*\.\s*)+[a-z]{2,24}(?![\w@])"
)
SIGNATURE_PATTERN = re.compile(
    r"(?is)\b(?P<signature>support\s+(?:team|equipment|staff)|destek\s+ekibi)\b.*$"
)
FOREIGN_GREETING_PATTERN = re.compile(
    r"(?i)^\s*(?:olá|ola|hola|bonjour|ciao|hallo|merhaba|buenos?\s+d[ií]as|boa\s+tarde)\s*[,!:.\-]*\s*"
)
FOREIGN_CONTENT_PATTERN = re.compile(
    r"(?i)\b(?:ja\s+tenim|però|estàs\s+utilitzant|remetrem|vostra\s+sol|"
    r"podem\s+afegir|equipo\s+de\s+apoyo|(?:apoio|apoyo)'?s\s+(?:equipe|team)|"
    r"si\s+necesita|háganos|haganos)\b"
)
SIGN_OFF_PATTERN = re.compile(
    r"(?is)\b(?:sincerely|regards|best\s+regards|kind\s+regards|greetings|"
    r"with\s+respect|summits|cumprimentos|atenciosamente|saudações|saudacoes|"
    r"saludos|atentamente)\b\s*[,!:.-]?.*$"
)
LOW_QUALITY_REPLY_PATTERN = re.compile(
    r"(?i)\b(?:your\s+problem\s+seems\s+to\s+be\s+solved|"
    r"support\s+(?:equipment|group)|our\s+respects|"
    r"we\s+can\s+call\s+the\s+solution)\b"
)


def keyword_support_group(message: str) -> str | None:
    """Açıkça ifade edilen yaygın destek konularını güvenli biçimde yönlendirir.

    Bu kurallar yalnızca güçlü konu kelimeleri içerir. Amaç LinearSVC'nin her
    girdiye zorunlu olarak bir sınıf atamasının, "şifre" gibi net bir talebi
    alakasız gruba götürmesini engellemektir; belirsiz metinlerde model çalışır.
    """
    text = re.sub(r"\s+", " ", message.casefold()).strip()
    terms = {
        "Account & data": (
            "password", "şifre", "sifre", "parola", "login", "log in", "sign in",
            "e-mail", "email address", "hesabım", "hesabim", "account",
        ),
        "Billing & subscriptions": (
            "payment", "pay ", "ödeme", "odeme", "credit", "kredi", "subscription",
            "subscribe", "abonelik", "aboneli", "refund", "iade", "invoice", "fatura",
            "pix", "oxxo", "efecty",
        ),
        # "Share a route" bir rota oluşturma isteği değil, paylaşım isteğidir.
        # Bu yüzden Route & navigation'dan önce değerlendirilir.
        "Sharing & collaboration": (
            "share", "sharing", "collaboration", "paylaş", "paylas", "ortak kullanıcı",
            "ortak kullanici",
        ),
        "Route & navigation": (
            "route", "navigation", "address", "stop", "rota", "navigasyon", "adres",
            "durak", "harita", "map",
        ),
        "Ads & promotions": (
            "advert", "watch ad", "video ad", "reklam", "video izle", "reward", "ödül", "odul",
        ),
    }
    for group, keywords in terms.items():
        if any(keyword in text for keyword in keywords):
            return group
    return None


def detect_language(text: str) -> str:
    """Destek veri kümesindeki diller arasından soru dilini algılar."""
    text = text.strip()
    if len(text) < 4:
        return "en"
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "en"
    if not candidates:
        return "en"
    language = candidates[0].lang.lower()
    return language if language in SUPPORTED_LANGUAGES else "en"


def load_chatbot(model_file: str | Path = DEFAULT_MODEL_FILE) -> dict:
    path = Path(model_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {path}. Önce 'python train_model.py' çalıştırın."
        )
    return joblib.load(path)


def clean_support_answer(answer: str, customer_message: str = "") -> str:
    """Önerilerden kişisel veri ve İngilizce dışı selamlama/imza kalıntılarını çıkarır."""
    # Destek imzasından sonraki kayıtlarda çoğunlukla ad, maskeli e-posta ve
    # kullanıcının tekrar edilmiş mesajı bulunur. Yalnızca gerçek destek yanıtını tut.
    signature = re.search(r"\n\s*Saygılarımızla,?\s*\n\s*Destek Ekibi\s*\n", answer, re.I)
    if signature:
        answer = answer[:signature.end()].strip()
    # Gövdesi İngilizce olup yalnızca selamlaması çevrilmemiş kayıtları standartlaştır.
    answer = FOREIGN_GREETING_PATTERN.sub("Hello, ", answer)
    # İmza, kişi adı, e-posta ve müşterinin kopyalanmış metni bu noktadan sonra gelir.
    if SIGN_OFF_PATTERN.search(answer):
        answer = SIGN_OFF_PATTERN.sub("", answer)
    # İngilizceye çevrilmiş kayıtlar çoğu zaman imza, ad ve e-postayı tek
    # satırda birleştirir. Support Team sonrasında hiçbir kişisel veri gösterme.
    answer = SIGNATURE_PATTERN.sub("", answer)
    # Normal veya yıldızla maskelenmiş bütün e-posta biçimlerini tamamen kaldır.
    answer = EMAIL_PATTERN.sub("", answer)
    if customer_message:
        answer = re.sub(
            rf"\n{{2,}}[^\n]*\n\s*{re.escape(customer_message.strip())}\s*$",
            "", answer, flags=re.I,
        )
    answer = re.sub(r"(?<=,)(?=[A-Za-z])", " ", answer)
    answer = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", answer)
    return re.sub(r"[ \t]{2,}", " ", answer).strip(" -–—,;|")


def is_safe_english_reply(answer: str) -> bool:
    """Yanıtın İngilizce öneri belleğinde gösterilmesi için güvenli olup olmadığını denetler."""
    cleaned = clean_support_answer(answer)
    # Bellek zaten İngilizce hazırlanmış veri içerir. Çeviriden kaldığı bilinen
    # yabancı ifade örnekleri deterministik olarak reddedilir; genel dil algılama
    # 9.000+ uzun yanıtta hem yavaş hem de karışık metinlerde daha az güvenilirdir.
    return (
        bool(cleaned)
        and not bool(FOREIGN_CONTENT_PATTERN.search(cleaned))
        and not bool(LOW_QUALITY_REPLY_PATTERN.search(cleaned))
    )


def retrieval_intent_boost(message: str, specific_category: str) -> float:
    """Açık destek niyetlerinde doğru alt konuyu genel kayıtlardan önce getirir."""
    query = message.casefold()
    category = specific_category.casefold()
    asks_how_to_pay = any(token in query for token in (
        "nasıl", "nasil", "how", "öd", "od", "pay", "pagar",
    ))
    if "pix" in query and asks_how_to_pay and "how to pay with pix" in category:
        return 0.25
    return 0.0


def retrieval_intent_category(message: str) -> str | None:
    """Net biçimde belirtilen özel destek konusunu döndürür."""
    query = message.casefold()
    asks_how_to_pay = any(token in query for token in (
        "nasıl", "nasil", "how", "öd", "od", "pay", "pagar",
    ))
    if "pix" in query and asks_how_to_pay:
        return "How to pay with PIX"
    return None


def clean_source_text(text: str) -> str:
    """Kaynak soruda kişisel e-posta görünmesini de engeller."""
    return re.sub(r"\s+", " ", EMAIL_PATTERN.sub("", text)).strip(" -–—,;|")


def simple_social_result(message: str, user_language: str) -> dict | None:
    """Kısa sohbet mesajlarını rastgele destek etiketine düşürmez."""
    normalized = re.sub(r"[^\wğüşöçıİĞÜŞÖÇа-яё]+", " ", message.casefold()).strip()
    words = normalized.split()
    phrase = " ".join(words)
    if len(words) <= 3 and (phrase in GREETING_WORDS or (words and words[0] in GREETING_WORDS)):
        category = "Welcome"
        answer = "Hello! How can we help you today?"
    elif len(words) <= 4 and (phrase in THANKS_WORDS or (words and words[0] in THANKS_WORDS)):
        category = "Thanks"
        answer = "You're welcome! Please let us know if you need any further help."
    elif len(words) <= 5 and (phrase in SOCIAL_PHRASES or (words and words[0] in SOCIAL_PHRASES)):
        category = "Social conversation"
        answer = "Hello! I am here to help with customer support questions. How can I assist you?"
    else:
        return None
    suggestion = {
        "answer": answer,
        "answer_en": answer,
        "matched_question": message,
        "matched_category": category,
        "similarity": 1.0,
        "match_type": "rule",
        "translation_available": True,
    }
    return {
        # Bu, model sınıfı değildir: kısa sosyal mesajlar açık kuralla ele alınır.
        # Arayüzde de "General inquiry" gibi yanıltıcı bir ana gruba dönüştürülmez.
        "category": category,
        "specific_category": category,
        "detected_language": user_language,
        "english_query": None,
        "memory_language": "en",
        **suggestion,
        "suggestions": [suggestion],
    }


def is_too_vague(message: str) -> bool:
    """Anlamsız ya da tek parçalı çok kısa girdiler için güvenli geri dönüş sağlar."""
    normalized = re.sub(r"\W+", "", message.casefold())
    if len(normalized) < 3:
        return True
    # aa, ???, zzz gibi tekrarlı kısa seslenişler destek talebi değildir.
    return len(normalized) <= 4 and len(set(normalized)) <= 2


def clarification_result(message: str, user_language: str, similarity: float) -> dict:
    """Veride yeterli eşleşme yoksa alakasız bir destek etiketi üretmez."""
    answer = "Could you please provide more details about the issue so we can help you?"
    suggestion = {
        "answer": answer,
        "answer_en": answer,
        "matched_question": message,
        "matched_category": "Needs clarification",
        "similarity": similarity,
        "match_type": "none",
        "translation_available": True,
    }
    return {
        "category": "Needs clarification",
        "specific_category": "Needs clarification",
        "detected_language": user_language,
        "english_query": None,
        "memory_language": "en",
        **suggestion,
        "suggestions": [suggestion],
    }


def answer_message(model: dict, message: str, top_k: int = 3) -> dict:
    """Çok dilli sorulardan İngilizce yanıt ve tutarlı kategori döndürür."""
    message = message.strip()
    if not message:
        raise ValueError("Mesaj boş olamaz.")

    english_memory = model.get("metadata", {}).get("memory_language") == "en"
    user_language = detect_language(message)
    social_result = simple_social_result(message, user_language)
    if social_result is not None:
        return social_result
    if is_too_vague(message):
        return clarification_result(message, user_language, 0.0)

    explicit_group = keyword_support_group(message)
    predicted_group = explicit_group or model["classifier"].predict([message])[0]
    query_vector = model["response_vectorizer"].transform([message])
    similarities = cosine_similarity(query_vector, model["response_matrix"]).ravel()
    # Öneriler sadece tahmin edilen ana gruptan seçilir; örneğin ödeme etiketi
    # altında rota yanıtı gösterilmez.
    group_indices = np.flatnonzero(
        np.asarray(model["training_categories"]) == predicted_group
    )
    if len(group_indices):
        group_similarities = np.full(len(similarities), -np.inf)
        group_similarities[group_indices] = similarities[group_indices]
        similarities = group_similarities
    top_k = max(1, min(top_k, len(similarities)))
    specific_categories = model.get("training_specific_categories", model["training_categories"])
    exact_intent = retrieval_intent_category(message)
    if exact_intent:
        exact_indices = np.flatnonzero(np.asarray(specific_categories) == exact_intent)
        if len(exact_indices):
            exact_similarities = np.full(len(similarities), -np.inf)
            exact_similarities[exact_indices] = similarities[exact_indices]
            similarities = exact_similarities
            # Aynı özel konu için yinelenen kayıtlar tek bir temiz taslakta
            # birleştirilir; kullanıcıya üç benzer varyant gösterilmez.
            top_k = 1
    boosted_scores = np.array(similarities, copy=True)
    for index in np.flatnonzero(np.isfinite(boosted_scores)):
        boosted_scores[index] += retrieval_intent_boost(message, specific_categories[int(index)])
    ranked_indices = np.argsort(boosted_scores)[::-1]
    suggestions = []
    seen_answers = set()
    for index in ranked_indices:
        if len(suggestions) >= top_k:
            break
        raw_answer = model["training_answers"][int(index)]
        if not is_safe_english_reply(raw_answer):
            continue
        english_answer = clean_support_answer(raw_answer, message)
        answer_key = re.sub(r"\s+", " ", english_answer.casefold()).strip()
        if answer_key in seen_answers:
            continue
        seen_answers.add(answer_key)
        suggestions.append({
            "answer": english_answer,
            "answer_en": english_answer,
            "matched_question": clean_source_text(model["training_texts"][int(index)]),
            "matched_category": model["training_categories"][int(index)],
            "matched_specific_category": specific_categories[int(index)],
            "similarity": float(similarities[int(index)]),
            "match_type": "retrieval",
            "translation_available": True,
        })
    if not suggestions:
        return clarification_result(message, user_language, 0.0)
    best = suggestions[0]
    if best["similarity"] < MIN_RETRIEVAL_SIMILARITY and not explicit_group:
        return clarification_result(message, user_language, best["similarity"])

    return {
        # Ana grup, doğrudan bu amaç için eğitilmiş LinearSVC sınıflandırıcısından
        # gelir; en yakın kayıt ise yalnızca İngilizce yanıt önerisini belirler.
        "category": predicted_group,
        "specific_category": best["matched_category"],
        "detected_language": user_language,
        "english_query": None,
        "memory_language": "en" if english_memory else "original-language",
        "category_source": "keyword_rule" if explicit_group else "model",
        "match_type": "retrieval",
        **best,
        "suggestions": suggestions,
    }
