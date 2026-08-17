"""Yerel Argos Translate ile dil algılama ve çeviri işlemleri.

Bu modül yalnızca çalışma anındaki çeviri katmanıdır. Müşteri destek modeli
ayrı olarak kendi verimizden eğitilmeye devam eder.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

# Argos'un indirilen paketlerini proje içinde tutuyoruz. Bu satırlar Argos import edilmeden önce çalışmalıdır; aksi halde kullanıcı dizinine yazmaya
# çalışır.
RUNTIME_DIR = Path(__file__).parent / ".translation_runtime"
os.environ.setdefault("XDG_DATA_HOME", str(RUNTIME_DIR / "data"))
os.environ.setdefault("XDG_CONFIG_HOME", str(RUNTIME_DIR / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_DIR / "cache"))
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
# Paketle gelen Stanza işaretleyicisi ek dil dosyaları isteyebilir. MiniSBD
# indirilen Argos paketleriyle çalışır ve ilk kurulumdan sonra offline kalır.
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")

import ctranslate2
from langdetect import DetectorFactory, LangDetectException, detect_langs
import argostranslate.package


DetectorFactory.seed = 42
MEMORY_LANGUAGE = "en"
# Veri kümesinde en çok görülen ve iki yönlü offline paketleri kurulan diller.
SUPPORTED_LANGUAGES = {
    "en", "tr", "pt", "es", "ru", "it", "fr", "de", "pl", "ro", "id", "ar", "uk", "nl"
}


def detect_language(text: str) -> str:
    """Metnin ISO dil kodunu döndürür; belirsiz/çok kısa metinde İngilizce seçer."""
    text = text.strip()
    if len(text) < 4:
        return MEMORY_LANGUAGE
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return MEMORY_LANGUAGE
    if not candidates:
        return MEMORY_LANGUAGE
    language = candidates[0].lang.lower()
    # langdetect'in zh-cn gibi kodlarını Argos/uygulama kodlarına dönüştür.
    language = {"zh-cn": "zh", "zh-tw": "zh"}.get(language, language)
    return language if language in SUPPORTED_LANGUAGES else MEMORY_LANGUAGE


def installed_language_codes() -> set[str]:
    codes = set()
    for package in argostranslate.package.get_installed_packages():
        if package.type == "translate":
            codes.update({package.from_code, package.to_code})
    return codes


@lru_cache(maxsize=None)
def _translation_resources(source: str, target: str):
    """Kurulu Argos paketinden doğrudan CTranslate2 kaynağını hazırlar.

    Argos'un varsayılan API'si cümle bölmek için ek Stanza/MiniSBD modelleri
    indirebilir. Burada kısa destek mesajlarını basit bir yerel bölücüyle
    işlediğimiz için ilk kurulumdan sonra hiçbir ağ çağrısı yapılmaz.
    """
    package = next(
        (
            item for item in argostranslate.package.get_installed_packages()
            if item.type == "translate" and item.from_code == source and item.to_code == target
        ),
        None,
    )
    if package is None:
        return None
    translator = ctranslate2.Translator(str(package.package_path / "model"), device="cpu")
    return package, translator


def is_translation_available(source: str, target: str) -> bool:
    if source == target:
        return True
    return _translation_resources(source, target) is not None


def translate_text(text: str, source: str, target: str) -> str:
    """Kurulu yerel paketle çevirir; uygun paket yoksa metni değiştirmez."""
    if not text.strip() or source == target:
        return text
    return translate_many([text], source, target)[0]


def translate_many(texts: list[str], source: str, target: str) -> list[str]:
    """Aynı dildeki metinleri toplu çevirerek eğitim hazırlığını hızlandırır."""
    if source == target:
        return texts.copy()
    resources = _translation_resources(source, target)
    if resources is None:
        return texts.copy()
    package, translator = resources

    chunk_counts = []
    chunks = []
    for text in texts:
        text_chunks = _split_translation_chunks(text)
        chunk_counts.append(len(text_chunks))
        chunks.extend(text_chunks)
    if not chunks:
        return texts.copy()

    pieces = []
    # Çok büyük bir batch bellek kullanımını artırıp ilk sonucu uzun süre
    # geciktirebilir. Küçük sabit partiler CPU üzerinde daha dengelidir.
    for start in range(0, len(chunks), 32):
        tokenized = [package.tokenizer.encode(chunk) for chunk in chunks[start:start + 32]]
        target_prefix = [[package.target_prefix]] * len(tokenized) if package.target_prefix else None
        translated = translator.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            replace_unknowns=True,
            max_batch_size=32,
            batch_type="tokens",
            beam_size=1,
            num_hypotheses=1,
            length_penalty=0.2,
        )
        for result in translated:
            value = package.tokenizer.decode(result.hypotheses[0])
            if package.target_prefix and value.startswith(package.target_prefix):
                value = value[len(package.target_prefix):]
            pieces.append(value.lstrip())

    translated_texts = []
    cursor = 0
    for original, count in zip(texts, chunk_counts):
        translated_texts.append(" ".join(pieces[cursor:cursor + count]) if count else original)
        cursor += count
    return translated_texts


def _split_translation_chunks(text: str, maximum_length: int = 280) -> list[str]:
    """Uzun/noktalamasız destek kayıtlarını çeviri için güvenli parçalara böler."""
    sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()]
    chunks = []
    for sentence in sentences:
        while len(sentence) > maximum_length:
            cut = sentence.rfind(" ", 0, maximum_length)
            if cut < maximum_length // 2:
                cut = maximum_length
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def to_english(text: str, source_language: str | None = None) -> tuple[str, str]:
    """Metni İngilizceye çevirir ve algılanan dili döndürür."""
    language = source_language or detect_language(text)
    return translate_text(text, language, MEMORY_LANGUAGE), language


def from_english(text: str, target_language: str) -> tuple[str, bool]:
    """İngilizce metni hedef dile çevirir; başarısızlıkta İngilizce döner."""
    if target_language == MEMORY_LANGUAGE:
        return text, True
    if not is_translation_available(MEMORY_LANGUAGE, target_language):
        return text, False
    return translate_text(text, MEMORY_LANGUAGE, target_language), True
