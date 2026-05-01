"""
=============================================================
ETAPA 1 - Preparacion y Limpieza del Dataset Bitext
=============================================================

Este script prepara el dataset Bitext Customer Support LLM
Chatbot Training Dataset para clasificacion de solicitudes.

Columnas usadas:
    instruction -> text
    category    -> category

No se usa la columna response, porque contiene la respuesta del
chatbot y no la solicitud original del cliente.
=============================================================
"""

import os
import re

import pandas as pd


# -------------------------------------------------------
# CONFIGURACION DE RUTAS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")

EXPECTED_RAW_PATH = os.path.join(
    RAW_DIR,
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
)

# Fallback para el nombre que existe actualmente en el workspace.
FALLBACK_RAW_PATH = os.path.join(RAW_DIR, "bitext_customer_support.csv")

CLEAN_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "tickets_clean.csv")
REQUIRED_COLUMNS = {"instruction", "category"}


def obtener_ruta_dataset() -> str:
    """Devuelve la ruta del dataset Bitext disponible."""
    if os.path.exists(EXPECTED_RAW_PATH):
        return EXPECTED_RAW_PATH
    if os.path.exists(FALLBACK_RAW_PATH):
        return FALLBACK_RAW_PATH
    raise FileNotFoundError(
        "No se encontro el dataset Bitext. Coloca el CSV en:\n"
        f"  {EXPECTED_RAW_PATH}\n"
        "o usa el nombre disponible actualmente:\n"
        f"  {FALLBACK_RAW_PATH}"
    )


def limpiar_placeholders(texto):
    """Elimina placeholders tipo {{Order Number}} y normaliza espacios."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"\{\{[^}]+\}\}", " ", texto)
    texto = re.sub(r"\{+[^}]+\}+", " ", texto)
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# -------------------------------------------------------
# PASO 1: Cargar dataset
# -------------------------------------------------------
print("=" * 60)
print("PASO 1: Cargando dataset Bitext...")
print("=" * 60)

raw_path = obtener_ruta_dataset()
df = pd.read_csv(raw_path)

print(f"  Ruta usada          : {raw_path}")
print(f"  Filas originales   : {len(df)}")
print(f"  Columnas detectadas: {list(df.columns)}")


# -------------------------------------------------------
# PASO 2: Validar columnas requeridas
# -------------------------------------------------------
print("\nPASO 2: Validando columnas requeridas...")
missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(
        "El dataset no contiene las columnas requeridas: "
        + ", ".join(sorted(missing))
    )
print("  Columnas requeridas encontradas: instruction, category")


# -------------------------------------------------------
# PASO 3: Construir dataset limpio
# -------------------------------------------------------
print("\nPASO 3: Construyendo text/category...")
df_clean = pd.DataFrame()
df_clean["text"] = df["instruction"].apply(limpiar_placeholders)
df_clean["category"] = df["category"].astype(str).str.strip().str.upper()


# -------------------------------------------------------
# PASO 4: Eliminar registros invalidos
# -------------------------------------------------------
print("\nPASO 4: Eliminando registros invalidos...")
antes = len(df_clean)

df_clean = df_clean[df_clean["text"].str.strip().str.len() >= 10]
df_clean = df_clean[df_clean["category"].str.strip().str.len() > 0]
df_clean = df_clean.drop_duplicates(subset=["text"])
df_clean = df_clean[["text", "category"]].reset_index(drop=True)

eliminados = antes - len(df_clean)
print(f"  Filas finales       : {len(df_clean)}")
print(f"  Registros eliminados: {eliminados}")


# -------------------------------------------------------
# PASO 5: Mostrar distribucion
# -------------------------------------------------------
print("\nPASO 5: Distribucion final por categoria:")
total = len(df_clean)
for cat, count in df_clean["category"].value_counts().sort_index().items():
    print(f"  {cat:<14}: {count:>5} ({count / total * 100:.1f}%)")

print(f"\n  Total de categorias detectadas: {df_clean['category'].nunique()}")


# -------------------------------------------------------
# PASO 6: Guardar
# -------------------------------------------------------
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8")

print(f"\nOK - Dataset guardado en: {CLEAN_PATH}")
