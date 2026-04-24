"""
=============================================================
ETAPA 1 - Preparación y Limpieza del Dataset
=============================================================

Este script:
1. Carga el dataset original de Kaggle
2. Explora su estructura
3. Mapea las categorías al español requerido
4. Limpia registros inválidos
5. Construye las columnas 'text' y 'category'
6. Guarda el dataset limpio en data/processed/tickets_clean.csv
"""

import pandas as pd
import re
import os

# -------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR,"data", "raw", "customer_support_tickets.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data" ,"processed", "tickets_clean.csv")

# -------------------------------------------------------
# PASO 1: Cargar el dataset
# -------------------------------------------------------
print("=" * 60)
print("PASO 1: Cargando dataset...")
print("=" * 60)

df = pd.read_csv(RAW_PATH)

print(f"  Filas totales    : {len(df)}")
print(f"  Columnas totales : {len(df.columns)}")
print(f"  Columnas         : {df.columns.tolist()}")

# -------------------------------------------------------
# PASO 2: Seleccionar columnas útiles
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 2: Seleccionando columnas relevantes...")
print("=" * 60)

# Columnas útiles para el proyecto:
# - Ticket Type  -> será nuestra etiqueta (category)
# - Ticket Subject -> parte del texto de entrada
# - Ticket Description -> parte del texto de entrada

columnas_utiles = ["Ticket Type", "Ticket Subject", "Ticket Description"]
df = df[columnas_utiles].copy()

print(f"  Columnas seleccionadas: {columnas_utiles}")

# -------------------------------------------------------
# PASO 3: Revisar valores únicos en la columna objetivo
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 3: Valores únicos en 'Ticket Type'...")
print("=" * 60)

print(df["Ticket Type"].value_counts().to_string())

# -------------------------------------------------------
# PASO 4: Mapeo de categorías al español
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 4: Mapeando categorías al español...")
print("=" * 60)

# Mapeo definido según los requerimientos del proyecto
CATEGORY_MAP = {
    "Technical issue"    : "Soporte Técnico",
    "Billing inquiry"    : "Facturación",
    "Product inquiry"    : "Consulta General",
    "Refund request"     : "Queja",
    "Cancellation request": "Cancelación"
}

df["category"] = df["Ticket Type"].map(CATEGORY_MAP)

# Verificar si quedó algún valor sin mapear
sin_mapear = df["category"].isna().sum()
print(f"  Registros sin mapear: {sin_mapear}")

if sin_mapear > 0:
    print("  Tipos no mapeados:")
    print(df[df["category"].isna()]["Ticket Type"].value_counts())

print("\n  Distribución de categorías mapeadas:")
print(df["category"].value_counts().to_string())

# -------------------------------------------------------
# PASO 5: Limpiar la columna de descripción
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 5: Limpiando texto de descripción...")
print("=" * 60)

def limpiar_descripcion(texto):
    """
    Elimina artefactos del dataset:
    - Placeholders como {product_purchased}
    - Espacios múltiples
    - Saltos de línea excesivos
    """
    if not isinstance(texto, str):
        return ""

    # Eliminar placeholders del tipo {variable}
    texto = re.sub(r'\{[^}]+\}', '', texto)

    # Eliminar saltos de línea y reemplazar por espacio
    texto = texto.replace('\n', ' ').replace('\r', ' ')

    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

df["Ticket Description"] = df["Ticket Description"].apply(limpiar_descripcion)
df["Ticket Subject"]     = df["Ticket Subject"].apply(
    lambda x: x.strip() if isinstance(x, str) else ""
)

print("  Limpieza aplicada: placeholders, saltos de línea, espacios extra.")

# -------------------------------------------------------
# PASO 6: Construir la columna 'text'
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 6: Construyendo columna 'text' (subject + description)...")
print("=" * 60)

df["text"] = df["Ticket Subject"] + " " + df["Ticket Description"]

# Ejemplo de texto construido
print("\n  Ejemplo de 'text' generado:")
print(f"  {df['text'].iloc[0][:200]}...")

# -------------------------------------------------------
# PASO 7: Eliminar registros inválidos
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 7: Eliminando registros inválidos...")
print("=" * 60)

antes = len(df)

# Eliminar filas con text vacío
df = df[df["text"].str.strip().str.len() > 10]

# Eliminar filas con category nula (si hubo algún tipo no mapeado)
df = df[df["category"].notna()]

# Eliminar duplicados exactos
df = df.drop_duplicates(subset=["text"])

despues = len(df)
print(f"  Registros antes  : {antes}")
print(f"  Registros después: {despues}")
print(f"  Eliminados       : {antes - despues}")

# -------------------------------------------------------
# PASO 8: Dataset final con solo columnas necesarias
# -------------------------------------------------------
df_clean = df[["text", "category"]].reset_index(drop=True)

# -------------------------------------------------------
# PASO 9: Verificar distribución final por categoría
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 9: Distribución final por categoría")
print("=" * 60)

conteo = df_clean["category"].value_counts()
total = len(df_clean)

for cat, count in conteo.items():
    porcentaje = (count / total) * 100
    print(f"  {cat:<20}: {count:>5} registros  ({porcentaje:.1f}%)")

print(f"\n  TOTAL: {total} registros")

# -------------------------------------------------------
# PASO 10: Guardar dataset limpio
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 10: Guardando dataset limpio...")
print("=" * 60)

os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8")

print(f"  Dataset guardado en: {CLEAN_PATH}")
print("\n✅ Etapa 1 completada exitosamente.")
print("   Puedes continuar con la Etapa 2: preprocessing.py")