"""Desteklenen diller için Argos Translate'in offline paketlerini indirir."""

from translation_service import MEMORY_LANGUAGE, SUPPORTED_LANGUAGES
import argostranslate.package


def main() -> None:
    print("📦 Argos Translate paket listesi güncelleniyor...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    required_pairs = [
        (language, MEMORY_LANGUAGE)
        for language in sorted(SUPPORTED_LANGUAGES - {MEMORY_LANGUAGE})
    ] + [
        (MEMORY_LANGUAGE, language)
        for language in sorted(SUPPORTED_LANGUAGES - {MEMORY_LANGUAGE})
    ]
    installed = {
        (package.from_code, package.to_code)
        for package in argostranslate.package.get_installed_packages()
    }

    for source, target in required_pairs:
        if (source, target) in installed:
            print(f"   ✓ {source} → {target} zaten kurulu")
            continue
        package = next(
            (item for item in available if item.from_code == source and item.to_code == target),
            None,
        )
        if package is None:
            print(f"   ! {source} → {target} paketi bulunamadı")
            continue
        print(f"   ↓ {source} → {target} indiriliyor...")
        argostranslate.package.install_from_path(package.download())

    print("✅ Offline çeviri paketleri hazır.")


if __name__ == "__main__":
    main()
