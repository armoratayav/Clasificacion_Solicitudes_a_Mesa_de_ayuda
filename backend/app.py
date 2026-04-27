"""
=============================================================
ETAPA 6 - Backend Web con Flask
=============================================================

Servidor Flask que expone el motor Naïve Bayes como API REST.

Endpoints:
    GET  /            → Estado del servidor
    GET  /health      → Health check con info del modelo
    POST /predict     → Clasificar un ticket

Uso:
    python3 backend/app.py

Ejemplo de request:
    POST http://localhost:5000/predict
    {
        "subject": "Problem with my invoice",
        "description": "I was charged twice this month"
    }

Ejemplo de response:
    {
        "ticket_id": "TCK-000123",
        "category": "Facturación",
        "scores": {
            "Facturación": 0.87,
            "Soporte Técnico": 0.06,
            ...
        }
    }
=============================================================
"""

import os
import sys
import random
import string
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# Agregar el directorio backend al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from naive_bayes import MultinomialNaiveBayes

# -------------------------------------------------------
# INICIALIZACIÓN DE FLASK
# -------------------------------------------------------
app = Flask(__name__)

# Habilitar CORS para que el frontend pueda hacer requests
# desde otro origen (ej: archivo HTML abierto directamente)
CORS(app)

# -------------------------------------------------------
# CARGA DEL MODELO AL INICIAR EL SERVIDOR
# -------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

print("=" * 55)
print("  Iniciando servidor Flask - Mesa de Ayuda IA")
print("=" * 55)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encontró el modelo en {MODEL_PATH}")
    print("Ejecuta primero: python3 backend/train.py")
    exit(1)

modelo = MultinomialNaiveBayes.load(MODEL_PATH)
print(f"  Modelo listo con {len(modelo.vocabulary)} palabras en vocabulario")
print(f"  Clases: {modelo.classes}")


# -------------------------------------------------------
# UTILIDADES
# -------------------------------------------------------

def generar_ticket_id() -> str:
    """
    Genera un ID de ticket único con formato TCK-XXXXXX.
    Combina timestamp y caracteres aleatorios para garantizar unicidad.
    """
    timestamp = datetime.now().strftime("%H%M%S")
    aleatorio = ''.join(random.choices(string.digits, k=4))
    return f"TCK-{timestamp}{aleatorio}"


def validar_entrada(data: dict) -> tuple:
    """
    Valida que el body del request tenga los campos requeridos
    y que no estén vacíos.

    Returns:
        (es_valido: bool, mensaje_error: str)
    """
    if not data:
        return False, "El body del request está vacío."

    subject     = data.get("subject", "").strip()
    description = data.get("description", "").strip()

    if not subject and not description:
        return False, "Debes proporcionar al menos 'subject' o 'description'."

    if len(subject) + len(description) < 5:
        return False, "El texto es demasiado corto para clasificar."

    return True, ""


# -------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Endpoint raíz — confirma que el servidor está activo.
    """
    return jsonify({
        "sistema" : "Sistema de Clasificación de Tickets - Mesa de Ayuda",
        "version" : "1.0.0",
        "estado"  : "activo",
        "endpoints": {
            "GET  /"        : "Este mensaje",
            "GET  /health"  : "Estado del modelo",
            "POST /predict" : "Clasificar un ticket"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check — devuelve información del modelo cargado.
    Útil para verificar que el servidor está listo.
    """
    return jsonify({
        "estado"     : "ok",
        "modelo"     : "MultinomialNaiveBayes",
        "clases"     : modelo.classes,
        "vocabulario": len(modelo.vocabulary),
        "timestamp"  : datetime.now().isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal de clasificación.

    Recibe un ticket con subject y description,
    devuelve la categoría predicha y los scores por clase.

    Request body (JSON):
        {
            "subject"     : "Problem with my invoice",
            "description" : "I was charged twice this month"
        }

    Response (JSON):
        {
            "ticket_id" : "TCK-143022-1234",
            "category"  : "Facturación",
            "scores"    : {
                "Facturación"    : 0.87,
                "Soporte Técnico": 0.06,
                "Consulta General": 0.04,
                "Queja"          : 0.02,
                "Cancelación"    : 0.01
            },
            "timestamp" : "2026-04-01T14:30:22"
        }
    """
    # --- Obtener y validar datos ---
    data     = request.get_json(silent=True)
    valido, error = validar_entrada(data)

    if not valido:
        return jsonify({
            "error": error,
            "ayuda": "Envía JSON con campos 'subject' y/o 'description'"
        }), 400

    # --- Construir texto de entrada ---
    subject     = data.get("subject", "").strip()
    description = data.get("description", "").strip()
    texto_input = f"{subject} {description}".strip()

    # --- Clasificar con el modelo ---
    try:
        categoria = modelo.predict(texto_input)
        scores    = modelo.predict_proba(texto_input)
    except Exception as e:
        return jsonify({
            "error"  : "Error al clasificar el ticket.",
            "detalle": str(e)
        }), 500

    # --- Construir respuesta ---
    respuesta = {
        "ticket_id": generar_ticket_id(),
        "category" : categoria,
        "scores"   : scores,
        "input"    : {
            "subject"    : subject,
            "description": description[:100] + "..." if len(description) > 100 else description
        },
        "timestamp": datetime.now().isoformat()
    }

    # Log en consola para seguimiento
    print(f"  [{respuesta['timestamp']}] "
          f"Ticket {respuesta['ticket_id']} -> {categoria} "
          f"(confianza: {scores[categoria]*100:.1f}%)")

    return jsonify(respuesta), 200


# -------------------------------------------------------
# MANEJO DE ERRORES GLOBALES
# -------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error"    : "Endpoint no encontrado.",
        "endpoints": ["GET /", "GET /health", "POST /predict"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Método HTTP no permitido para este endpoint."
    }), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Error interno del servidor.",
        "detalle": str(e)
    }), 500


# -------------------------------------------------------
# INICIO DEL SERVIDOR
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n  Servidor corriendo en: http://localhost:5000")
    print("  Presiona Ctrl+C para detener\n")
    print("=" * 55)

    app.run(
        host="0.0.0.0",   # Acepta conexiones desde cualquier IP
        port=5000,
        debug=False        # False en producción
    )
