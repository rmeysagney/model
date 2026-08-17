"""Veri dosyalarını okumak için ortak yardımcılar."""

import json
import re
from pathlib import Path


CATEGORY_PREFIX = re.compile(r"^\[Category:\s*(.*?)\]\s*", re.DOTALL)

# Aynı desteği ifade eden küçük etiket farklarını tek sınıfta toplar. Böylece
# "requirement" ve "requirements" gibi yazımlar ayrı model sınıfları olmaz.
CATEGORY_ALIASES = {
    "requirement": "Requirement",
    "requirements": "Requirement",
    "feature request": "Feature request",
    "feature requests": "Feature request",
    "feature request (walk)": "Feature request",
    "payment problem": "Payment problem",
    "payment problem (seems to be resolved)": "Payment problem",
    "payment problem (brazil)": "Payment problem",
    "payment problem (turkey)": "Payment problem",
    "payment options": "Payment options",
    "payment options (argentina)": "Payment options",
    "payment options (brazil)": "Payment options",
    "payment options (colombia)": "Payment options",
    "problem": "Problem",
    "problem?": "Problem",
    "problem ?": "Problem",
    "cancel subscription": "Cancel subscription",
    "how to cancel subscription": "Cancel subscription",
    "subscription cancellation request": "Cancel subscription",
    "subscription cancel request": "Cancel subscription",
    "change payment method": "Change payment method",
    "how to change payment method": "Change payment method",
    "address is wrong": "Address is wrong",
    "address wrong": "Address is wrong",
    "error message requested": "Error message or screenshot requested",
    "screenshot or screen video requested": "Error message or screenshot requested",
    "screen video requested": "Error message or screenshot requested",
}


def canonicalize_category(value: str) -> str:
    """Etiketi okunabilir, tekil ve tutarlı bir sınıf adına dönüştürür."""
    label = re.sub(r"\s+", " ", str(value or "").strip())
    if not label:
        return "General"
    key = label.casefold()
    return CATEGORY_ALIASES.get(key, label)


def group_category(specific_category: str) -> str:
    """İnce taneli destek etiketlerini iş açısından anlamlı ana gruplara toplar."""
    label = canonicalize_category(specific_category).casefold()
    special_labels = {
        "welcome": "Welcome",
        "thanks": "Thanks",
        "social conversation": "Social conversation",
    }
    if label in special_labels:
        return special_labels[label]
    if "requirement" in label:
        return "Requirement"
    if "feature request" in label:
        return "Feature request"
    if any(word in label for word in (
        "payment", "credit", "subscription", "subscribe", "refund", "invoice", "price", "pay with",
        "free or paid", "free usage", "free credit", "gift credit", "gift", "currency",
        "charge", "pix", "efecty", "oxxo",
    )):
        return "Billing & subscriptions"
    if any(word in label for word in ("route", "stop", "address", "navigation", "map", "geocode")):
        return "Route & navigation"
    if any(word in label for word in (
        "account", "email", "password", "device", "backup", "data lost", "login", "logout",
    )):
        return "Account & data"
    if any(word in label for word in ("share", "collaboration", "another user", "whatsapp")):
        return "Sharing & collaboration"
    if any(word in label for word in ("ad", "video", "reward", "earn")):
        return "Ads & promotions"
    if any(word in label for word in (
        "first usage", "information", "what is", "how to", "manual", "excel", "time estimation",
        "time window", "sort", "reorder", "fast input", "beta",
    )):
        return "Getting started & information"
    if any(word in label for word in (
        "not for companies", "web or desktop", "restriction", "huawei", "another country", "availability",
    )):
        return "Access & availability"
    if any(word in label for word in ("phone", "team", "support request", "replied via")):
        return "Contact support"
    if any(word in label for word in ("error", "bug", "not working", "update", "problem")):
        return "Technical issue"
    return "General inquiry"


def normalise_record(item: dict) -> dict:
    """Bir kaydı ``text``, ``category`` ve ``answer`` alanlarına dönüştürür.

    Yeni veri biçimi doğrudan bu üç alanı kullanır. Önceki çalışmadan kalan
    sohbet-biçimli JSONL dosyaları varsa, yeniden veri hazırlamadan okunabilsin
    diye yalnızca bu fonksiyonda geriye dönük destek bulunur.
    """
    if {"text", "category", "answer"}.issubset(item):
        return {
            "text": str(item["text"]).strip(),
            "category": canonicalize_category(item["category"]),
            "answer": str(item["answer"]).strip(),
        }

    messages = item.get("messages", [])
    if len(messages) < 3:
        raise ValueError("Kayıtta text/category/answer alanları bulunmuyor.")

    text = str(messages[1].get("content", "")).strip()
    assistant_text = str(messages[2].get("content", "")).strip()
    match = CATEGORY_PREFIX.match(assistant_text)
    # Ham veri bazen birden fazla etiketi virgülle birleştiriyor. Chatbot tek kategori gösterdiği için ilk etiket, yani kaydın ana destek konusu kullanılır.
    raw_category = match.group(1).strip() if match else "General"
    category = canonicalize_category(raw_category.split(",", 1)[0])
    answer = assistant_text[match.end():].strip() if match else assistant_text
    return {"text": text, "category": category, "answer": answer}


def load_records(path: str | Path) -> list[dict]:
    """JSONL dosyasını okuyup boş veya geçersiz satırları atar."""
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = normalise_record(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number} okunamadı: {error}") from error
            if record["text"] and record["answer"]:
                records.append(record)
    return records


def save_records(records: list[dict], path: str | Path) -> None:
    """Yalın eğitim veri biçimini JSONL olarak kaydeder."""
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
