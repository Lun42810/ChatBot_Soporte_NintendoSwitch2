import os

import groq
import re


from dotenv import load_dotenv

load_dotenv()  # antes de importar rag, para que lea la configuración del .env

from flask import Flask, jsonify, render_template, request  # noqa: E402

from rag import MAX_HISTORY_TURNS, RAGConfig, RAGError, answer_question, get_vector_store  # noqa: E402

app = Flask(__name__)

# Precarga el índice (y el modelo de embeddings) al arrancar, para que la primera pregunta no tarde.
try:
    get_vector_store(RAGConfig.from_env())
except Exception:
    app.logger.exception("No se pudo precargar el índice")


def _clean_history(raw):
    """Valida el historial enviado por el cliente y conserva solo los últimos turnos."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("'history' debe ser una lista.")
    cleaned = []
    for item in raw[-MAX_HISTORY_TURNS:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            cleaned.append({"role": item["role"], "content": item["content"][:2000]})
    return cleaned


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "El cuerpo debe ser un objeto JSON con el campo 'message'."}), 400

    question = data.get("message")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "El campo 'message' es obligatorio y debe ser texto."}), 400
    question = question.strip()[:1000]

    try:
        history = _clean_history(data.get("history"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        result = answer_question(question, history)
    except RAGError as error:
        return jsonify({"error": str(error)}), 503
    except groq.RateLimitError:
        return jsonify({"error": "El servicio de IA alcanzó su límite de uso. Intenta de nuevo en unos minutos."}), 429
    except Exception:
        app.logger.exception("Error al procesar la pregunta mediante RAG")
        return jsonify({"error": "No fue posible procesar la pregunta. Revisa la configuración y los registros del servidor."}), 500

    return jsonify({
        "question": question,
        "response": re.sub(r"\*\*(.+?)\*\*", r"\1", result["answer"]),
        "sources": result["sources"],
        "standalone_question": result["standalone_question"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
