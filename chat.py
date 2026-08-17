"""Eğitilmiş, sıfırdan oluşturulmuş müşteri destek modeliyle sohbet."""

from chatbot_model import answer_message, load_chatbot


def main() -> None:
    print("📥 Model yükleniyor...")
    model = load_chatbot()
    metadata = model["metadata"]
    print(
        f"✅ Model hazır ({metadata['training_examples']} kayıt, "
        f"{metadata['category_count']} kategori). Çıkmak için 'q' yazın."
    )
    print("─" * 60)

    while True:
        try:
            message = input("\n👤 Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz! 👋")
            return

        if message.lower() in {"q", "quit", "exit", "çıkış"}:
            print("Görüşürüz! 👋")
            return
        if not message:
            continue

        result = answer_message(model, message)
        print(f"\n🏷️  Kategori: {result['category']}")
        print(f"🤖 Yanıt: {result['answer']}")
        print(f"\n🔎 Benzerlik: %{result['similarity'] * 100:.1f}")
        print(f"   Eşleşen kayıt: {result['matched_question'][:160]}")
        print("─" * 60)


if __name__ == "__main__":
    main()
