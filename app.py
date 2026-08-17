"""Tarayıcı tabanlı müşteri destek chatbot arayüzü."""

import os

from flask import Flask, jsonify, render_template, request

from chatbot_model import answer_message, load_chatbot


app = Flask(__name__)
model = None


def get_model():
    """Büyük model dosyasını yalnızca ilk istek geldiğinde belleğe alır."""
    global model
    if model is None:
        model = load_chatbot()
    return model


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    """Sunucunun ayakta olduğunu doğrulayan hafif yayın sağlık kontrolü."""
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    try:
        chatbot = get_model()
        return jsonify({"ready": True, "metadata": chatbot["metadata"]})
    except FileNotFoundError as error:
        return jsonify({"ready": False, "error": str(error)}), 503
    except Exception:
        app.logger.exception("Model status could not be loaded")
        return jsonify({"ready": False, "error": "Model yüklenirken beklenmeyen bir hata oluştu."}), 500


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Lütfen bir mesaj yazın."}), 400
    if len(message) > 5_000:
        return jsonify({"error": "Mesaj en fazla 5.000 karakter olabilir."}), 400

    try:
        result = answer_message(get_model(), message, top_k=3)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Chat analysis failed")
        return jsonify({"error": "Analiz sırasında beklenmeyen bir hata oluştu."}), 500
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"\n🌐 Arayüz hazır: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
