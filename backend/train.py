"""
=============================================================
ETAPA 5 - Entrenamiento Final y Persistencia del Modelo
=============================================================

Este script:
    1. Carga el dataset limpio completo
    2. Entrena el modelo final con el 100% de los datos
    3. Guarda el modelo entrenado en backend/model.pkl
    4. Verifica que el modelo guardado carga y predice correctamente

A diferencia de evaluate.py (que divide en folds para medir
rendimiento), aquí usamos TODO el dataset para maximizar
el vocabulario y la cantidad de ejemplos por clase.
=============================================================
"""

import os
import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from naive_bayes import MultinomialNaiveBayes

# -------------------------------------------------------
# RUTAS
# -------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "..", "data", "processed", "tickets_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# -------------------------------------------------------
# PASO 1: Cargar dataset completo
# -------------------------------------------------------
print("=" * 60)
print("ENTRENAMIENTO FINAL DEL MODELO")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    print("ERROR: No se encontró tickets_clean.csv")
    print("Ejecuta primero: python3 backend/prepare_dataset.py")
    exit(1)

df = pd.read_csv(CSV_PATH)
print(f"\nDataset cargado: {len(df)} registros")
print("\nDistribución por categoría:")
for cat, count in df["category"].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {cat:<22}: {count:>5} ({pct:.1f}%)")

texts  = df["text"].tolist()
labels = df["category"].tolist()

# -------------------------------------------------------
# PASO 2: Entrenar con el 100% del dataset
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Entrenando modelo con 100% del dataset...")
print("=" * 60)

inicio = time.time()
modelo = MultinomialNaiveBayes()
modelo.fit(texts, labels)
tiempo = time.time() - inicio

print(f"\n  Tiempo de entrenamiento: {tiempo:.2f} segundos")

# -------------------------------------------------------
# PASO 3: Guardar el modelo
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Guardando modelo en model.pkl...")
print("=" * 60)

modelo.save(MODEL_PATH)

# -------------------------------------------------------
# PASO 4: Verificar carga y predicción
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Verificando carga del modelo...")
print("=" * 60)

modelo_cargado = MultinomialNaiveBayes.load(MODEL_PATH)

# Casos de prueba de verificación
casos = [
    ("billing charged twice invoice payment error",       "Facturación"),
    ("technical issue software crash error device",       "Soporte Técnico"),
    ("cancel subscription account closure terminate",     "Cancelación"),
    ("product features information pricing inquiry",      "Consulta General"),
    ("complaint dissatisfied poor service quality",       "Queja"),
]

print("\n  Verificación de predicciones:")
print(f"  {'Texto':<45} {'Esperado':<22} {'Predicho':<22} OK")
print(f"  {'-'*100}")

todos_correctos = True
for texto, esperado in casos:
    predicho = modelo_cargado.predict(texto)
    ok       = "OK" if predicho == esperado else "ERROR"
    if predicho != esperado:
        todos_correctos = False
    print(f"  {texto:<45} {esperado:<22} {predicho:<22} {ok}")

# -------------------------------------------------------
# RESUMEN FINAL
# -------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN DEL MODELO FINAL")
print("=" * 60)
print(f"  Archivo         : {MODEL_PATH}")
print(f"  Tamaño          : {os.path.getsize(MODEL_PATH)/1024:.1f} KB")
print(f"  Clases          : {modelo_cargado.classes}")
print(f"  Vocabulario     : {len(modelo_cargado.vocabulary)} palabras")
print(f"  Total documentos: {len(texts)}")
print(f"  Verificación    : {'OK - TODAS CORRECTAS' if todos_correctos else 'ERROR - HAY ERRORES'}")

print("\nOK - Etapa 5 completada.")
print("   El modelo está listo en: backend/model.pkl")
print("   Puedes continuar con la Etapa 6: app.py (Flask)")
