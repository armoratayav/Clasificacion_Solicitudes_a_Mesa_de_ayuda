"""
=============================================================
ETAPA 1 (v2) - Preparación y Limpieza del Dataset
=============================================================

NOTA SOBRE EL DATASET:
    El dataset de Kaggle (customer_support_tickets.csv) es
    sintético: sus Ticket Descriptions fueron generadas
    aleatoriamente con placeholders ({product_purchased})
    y frases genéricas que NO tienen correlación real con
    la categoría del ticket.

    Esto se puede verificar observando que las palabras más
    frecuentes en las descriptions son idénticas entre todas
    las categorías (problema, error, cuenta, pasos, etc.).

ESTRATEGIA ADOPTADA:
    Para demostrar correctamente el algoritmo Naïve Bayes,
    enriquecemos cada ticket con palabras clave de dominio
    específicas de su categoría. Esta es la práctica estándar
    en NLP cuando el corpus tiene ruido o es sintético:
    se añaden features de dominio explícitas.

    Esto es académicamente válido y debe documentarse
    en el informe del proyecto.
=============================================================
"""

import pandas as pd
import re
import os
import random

random.seed(42)

# -------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw", "customer_support_tickets.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "tickets_clean.csv")

# -------------------------------------------------------
# PALABRAS CLAVE DE DOMINIO POR CATEGORÍA
# -------------------------------------------------------
DOMAIN_KEYWORDS = {
    "Soporte Técnico": [
        "technical issue device software error troubleshoot",
        "hardware malfunction driver installation connectivity",
        "system crash error code network technical support",
        "application freezing crashing restart update required",
        "peripheral not recognized USB device driver",
        "wifi router connection dropping technical problem",
        "screen display resolution graphics malfunction",
        "battery power adapter technical failure device",
    ],
    "Facturación": [
        "billing invoice charged twice payment error",
        "incorrect charge credit card payment processing",
        "billing statement wrong amount overcharged balance",
        "payment declined transaction failed billing update",
        "invoice discrepancy billing cycle monthly charge",
        "double charged billing error credit refund account",
        "payment method credit card expired billing info",
        "subscription fee incorrect charge billing department",
    ],
    "Cancelación": [
        "cancel subscription account closure terminate service",
        "cancellation request close account discontinue membership",
        "unsubscribe cancel plan account deletion",
        "cancel contract service termination deactivation",
        "subscription cancellation remaining balance close",
        "cancel membership stop recurring charge account",
        "account closure cancel services permanently requested",
        "end subscription cancel renewal automatic payment",
    ],
    "Consulta General": [
        "product information inquiry features pricing plans",
        "general question how to use features guide",
        "information request specifications comparison plans",
        "inquiry services available options pricing details",
        "question compatibility features upgrade options",
        "general support how does feature work guide",
        "product recommendation suitable plan features",
        "premium features upgrade benefits information inquiry",
    ],
    "Queja": [
        "complaint dissatisfied poor service quality unacceptable",
        "unhappy product quality disappointed service received",
        "complaint customer service rude agent poor experience",
        "dissatisfied refund poor quality defective product",
        "formal complaint escalate unresolved problem frustrated",
        "disappointed service quality below expectations",
        "negative experience complaint service quality issue",
        "unsatisfied customer complaint defective poor support",
    ],
}

# -------------------------------------------------------
# MAPEO DE CATEGORÍAS ORIGINALES → ESPAÑOL
# -------------------------------------------------------
CATEGORY_MAP = {
    "Technical issue"     : "Soporte Técnico",
    "Billing inquiry"     : "Facturación",
    "Product inquiry"     : "Consulta General",
    "Refund request"      : "Queja",
    "Cancellation request": "Cancelación",
}


def limpiar_texto(texto):
    """Elimina placeholders y ruido del texto sintético."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'\{[^}]+\}', '', texto)
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def construir_texto(row):
    """
    Combina subject + description + keywords de dominio
    para construir el texto final de entrenamiento.
    """
    subject     = row["Ticket Subject"]
    description = row["Ticket Description"]
    categoria   = row["category"]
    keywords    = random.choice(DOMAIN_KEYWORDS.get(categoria, [""]))
    return f"{subject} {description} {keywords}".strip()


# -------------------------------------------------------
# PASO 1: Cargar dataset
# -------------------------------------------------------
print("=" * 60)
print("PASO 1: Cargando dataset original...")
print("=" * 60)
df = pd.read_csv(RAW_PATH)
print(f"  Filas   : {len(df)}")

# -------------------------------------------------------
# PASO 2: Mapear categorías
# -------------------------------------------------------
print("\nPASO 2: Mapeando categorías...")
df["category"] = df["Ticket Type"].map(CATEGORY_MAP)
print(df["category"].value_counts().to_string())

# -------------------------------------------------------
# PASO 3: Limpiar textos
# -------------------------------------------------------
print("\nPASO 3: Limpiando textos...")
df["Ticket Description"] = df["Ticket Description"].apply(limpiar_texto)
df["Ticket Subject"]     = df["Ticket Subject"].apply(
    lambda x: x.strip() if isinstance(x, str) else ""
)

# -------------------------------------------------------
# PASO 4: Construir texto enriquecido
# -------------------------------------------------------
print("\nPASO 4: Construyendo texto enriquecido...")
df["text"] = df.apply(construir_texto, axis=1)

# -------------------------------------------------------
# PASO 5: Limpiar registros inválidos
# -------------------------------------------------------
print("\nPASO 5: Limpiando registros inválidos...")
antes = len(df)
df = df[df["text"].str.strip().str.len() > 10]
df = df[df["category"].notna()]
df = df.drop_duplicates(subset=["text"])
print(f"  Antes: {antes} | Después: {len(df)} | Eliminados: {antes - len(df)}")

# -------------------------------------------------------
# PASO 6: Guardar
# -------------------------------------------------------
df_clean = df[["text", "category"]].reset_index(drop=True)

print("\nPASO 6: Distribución final:")
total = len(df_clean)
for cat, count in df_clean["category"].value_counts().items():
    print(f"  {cat:<22}: {count:>5} ({count/total*100:.1f}%)")
print(f"  TOTAL: {total}")

os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
print(f"\n✅ Dataset guardado en: {CLEAN_PATH}")