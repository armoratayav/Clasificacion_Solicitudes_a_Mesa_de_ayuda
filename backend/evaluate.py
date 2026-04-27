"""
=============================================================
ETAPA 4 - Evaluación con K-Folds Cross Validation
=============================================================

Implementación DESDE CERO de:
    - K-Folds Cross Validation (K=5)
    - Matriz de Confusión
    - Accuracy
    - Precisión por clase
    - Recall por clase
    - F1-Score por clase
    - Macro F1
    - Promedio y varianza entre folds

Fórmulas utilizadas:
--------------------
Accuracy   = correctos / total

Precisión  = TP / (TP + FP)
    De todos los que predije como clase C,
    ¿cuántos realmente eran clase C?

Recall     = TP / (TP + FN)
    De todos los que realmente eran clase C,
    ¿cuántos predije correctamente?

F1-Score   = 2 * (Precisión * Recall) / (Precisión + Recall)
    Media armónica entre Precisión y Recall.

Macro F1   = promedio de F1 de todas las clases (sin ponderar)
=============================================================
"""

import os
import sys
import math
import pandas as pd
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from naive_bayes import MultinomialNaiveBayes


# ============================================================
# MÉTRICAS DESDE CERO
# ============================================================

def calcular_matriz_confusion(y_true: list, y_pred: list, clases: list) -> dict:
    """
    Construye la matriz de confusión manualmente.

    La matriz de confusión es una tabla donde:
        - Filas    = clase real
        - Columnas = clase predicha
        - Celda[i][j] = cuántas veces la clase i fue predicha como j

    Args:
        y_true  : etiquetas reales
        y_pred  : etiquetas predichas
        clases  : lista de clases únicas

    Returns:
        dict: {clase_real: {clase_pred: conteo}}
    """
    matriz = {c: {p: 0 for p in clases} for c in clases}
    for real, pred in zip(y_true, y_pred):
        if real in matriz and pred in matriz[real]:
            matriz[real][pred] += 1
    return matriz


def calcular_metricas_por_clase(matriz: dict, clases: list) -> dict:
    """
    Calcula Precisión, Recall y F1-Score por clase
    a partir de la matriz de confusión.

    Para cada clase C:
        TP = matriz[C][C]           (predicho C y era C)
        FP = sum(matriz[X][C]) - TP (predicho C pero era otra)
        FN = sum(matriz[C][X]) - TP (era C pero predicho otra)

    Args:
        matriz : matriz de confusión
        clases : lista de clases

    Returns:
        dict: {clase: {precision, recall, f1}}
    """
    metricas = {}

    for clase in clases:
        # Verdaderos positivos: diagonal de la matriz
        TP = matriz[clase][clase]

        # Falsos positivos: columna de clase - TP
        FP = sum(matriz[c][clase] for c in clases) - TP

        # Falsos negativos: fila de clase - TP
        FN = sum(matriz[clase][c] for c in clases) - TP

        # Precisión
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        # Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        # F1-Score (media armónica)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        metricas[clase] = {
            "TP"       : TP,
            "FP"       : FP,
            "FN"       : FN,
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "f1"       : round(f1, 4),
        }

    return metricas


def calcular_accuracy(y_true: list, y_pred: list) -> float:
    """
    Accuracy = número de predicciones correctas / total.

    Args:
        y_true : etiquetas reales
        y_pred : etiquetas predichas

    Returns:
        float: accuracy entre 0 y 1
    """
    correctos = sum(1 for r, p in zip(y_true, y_pred) if r == p)
    return correctos / len(y_true) if len(y_true) > 0 else 0.0


def calcular_macro_f1(metricas_por_clase: dict) -> float:
    """
    Macro F1 = promedio simple de F1 de todas las clases.
    No pondera por cantidad de ejemplos por clase.

    Args:
        metricas_por_clase: resultado de calcular_metricas_por_clase()

    Returns:
        float: Macro F1
    """
    f1_scores = [m["f1"] for m in metricas_por_clase.values()]
    return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0


def calcular_varianza(valores: list) -> float:
    """
    Varianza poblacional: mide qué tan estables son los resultados
    entre folds. Varianza baja = modelo estable y generalizable.

    Varianza = (1/n) * SUM (xi - media)^2

    Args:
        valores: lista de métricas por fold

    Returns:
        float: varianza
    """
    if len(valores) < 2:
        return 0.0
    media = sum(valores) / len(valores)
    return round(sum((x - media) ** 2 for x in valores) / len(valores), 6)


# ============================================================
# K-FOLDS CROSS VALIDATION MANUAL
# ============================================================

def kfolds_split(n: int, k: int, seed: int = 42) -> list:
    """
    Divide los índices de 0..n-1 en K particiones balanceadas.

    Algoritmo:
        1. Crear lista de índices [0, 1, 2, ..., n-1]
        2. Mezclarlos aleatoriamente (Fisher-Yates shuffle)
        3. Dividir en K partes aproximadamente iguales

    Args:
        n    : número total de muestras
        k    : número de folds
        seed : semilla para reproducibilidad

    Returns:
        list: K listas de índices, una por fold
    """
    # Mezcla determinista con semilla
    indices = list(range(n))

    # Fisher-Yates shuffle manual
    import random
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Dividir en K partes
    fold_size = n // k
    folds = []
    for i in range(k):
        inicio = i * fold_size
        fin    = inicio + fold_size if i < k - 1 else n
        folds.append(indices[inicio:fin])

    return folds


def evaluar_kfolds(texts: list, labels: list, k: int = 5) -> dict:
    """
    Ejecuta K-Folds Cross Validation completo.

    En cada iteración:
        - 1 fold = conjunto de validación
        - K-1 folds = conjunto de entrenamiento
        - Se entrena un modelo nuevo y se evalúa

    Args:
        texts  : lista de textos
        labels : lista de etiquetas
        k      : número de folds (default=5)

    Returns:
        dict con resultados por fold y promedios finales
    """
    clases   = sorted(list(set(labels)))
    folds    = kfolds_split(len(texts), k)

    # Resultados por fold
    resultados_folds = []

    # Matriz de confusión acumulada (para reporte final)
    matriz_global    = {c: {p: 0 for p in clases} for c in clases}

    print(f"\n{'='*60}")
    print(f"K-FOLDS CROSS VALIDATION  (K={k})")
    print(f"{'='*60}")
    print(f"Total muestras : {len(texts)}")
    print(f"Clases         : {clases}")
    print(f"Tamaño de fold : ~{len(texts)//k} muestras")

    for fold_idx in range(k):
        print(f"\n{'─'*60}")
        print(f"FOLD {fold_idx + 1} / {k}")
        print(f"{'─'*60}")

        # --- Definir índices de validación y entrenamiento ---
        val_indices   = folds[fold_idx]
        train_indices = []
        for j in range(k):
            if j != fold_idx:
                train_indices.extend(folds[j])

        # --- Construir conjuntos ---
        train_texts  = [texts[i]  for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts    = [texts[i]  for i in val_indices]
        val_labels   = [labels[i] for i in val_indices]

        print(f"  Entrenamiento : {len(train_texts)} muestras")
        print(f"  Validación    : {len(val_texts)} muestras")

        # --- Entrenar modelo ---
        modelo = MultinomialNaiveBayes()
        modelo.fit(train_texts, train_labels)

        # --- Predecir ---
        predicciones = [modelo.predict(t) for t in val_texts]

        # --- Calcular métricas ---
        accuracy      = calcular_accuracy(val_labels, predicciones)
        matriz        = calcular_matriz_confusion(val_labels, predicciones, clases)
        metricas_clase = calcular_metricas_por_clase(matriz, clases)
        macro_f1      = calcular_macro_f1(metricas_clase)

        # Acumular en matriz global
        for real in clases:
            for pred in clases:
                matriz_global[real][pred] += matriz[real][pred]

        # --- Mostrar resultados del fold ---
        print(f"\n  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Macro F1  : {macro_f1:.4f}")
        print(f"\n  {'Clase':<22} {'Precisión':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*55}")
        for clase in clases:
            m = metricas_clase[clase]
            print(f"  {clase:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

        # Guardar resultados del fold
        resultados_folds.append({
            "fold"          : fold_idx + 1,
            "accuracy"      : accuracy,
            "macro_f1"      : macro_f1,
            "metricas_clase": metricas_clase,
            "matriz"        : matriz,
        })

    # --------------------------------------------------------
    # PROMEDIOS Y VARIANZA ENTRE FOLDS
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("RESUMEN FINAL - PROMEDIOS ENTRE FOLDS")
    print(f"{'='*60}")

    accuracies = [r["accuracy"] for r in resultados_folds]
    macro_f1s  = [r["macro_f1"]  for r in resultados_folds]

    acc_prom   = round(sum(accuracies) / k, 4)
    f1_prom    = round(sum(macro_f1s)  / k, 4)
    acc_var    = calcular_varianza(accuracies)
    f1_var     = calcular_varianza(macro_f1s)

    print(f"\n  Accuracy  promedio : {acc_prom:.4f}  ({acc_prom*100:.2f}%)")
    print(f"  Accuracy  varianza : {acc_var:.6f}")
    print(f"  Macro F1  promedio : {f1_prom:.4f}")
    print(f"  Macro F1  varianza : {f1_var:.6f}")

    # Promediar métricas por clase entre todos los folds
    print(f"\n  {'Clase':<22} {'Prec. Prom':>10} {'Rec. Prom':>10} {'F1 Prom':>10}")
    print(f"  {'─'*55}")

    metricas_promedio = {}
    clases_metricas = list(resultados_folds[0]["metricas_clase"].keys())

    for clase in clases_metricas:
        prec_vals = [r["metricas_clase"][clase]["precision"] for r in resultados_folds]
        rec_vals  = [r["metricas_clase"][clase]["recall"]    for r in resultados_folds]
        f1_vals   = [r["metricas_clase"][clase]["f1"]        for r in resultados_folds]

        prec_prom = round(sum(prec_vals) / k, 4)
        rec_prom  = round(sum(rec_vals)  / k, 4)
        f1_prom_c = round(sum(f1_vals)   / k, 4)

        metricas_promedio[clase] = {
            "precision": prec_prom,
            "recall"   : rec_prom,
            "f1"       : f1_prom_c,
        }
        print(f"  {clase:<22} {prec_prom:>10.4f} {rec_prom:>10.4f} {f1_prom_c:>10.4f}")

    # --------------------------------------------------------
    # MATRIZ DE CONFUSIÓN GLOBAL (acumulada de todos los folds)
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("MATRIZ DE CONFUSIÓN GLOBAL (acumulada)")
    print(f"{'='*60}")
    print("  (Filas = Real | Columnas = Predicho)\n")

    # Encabezado abreviado
    abrev = {c: c[:8] for c in clases}
    header = "  " + " " * 22
    for c in clases:
        header += f"{abrev[c]:>10}"
    print(header)
    print("  " + "─" * (22 + 10 * len(clases)))

    for real in clases:
        fila = f"  {real:<22}"
        for pred in clases:
            val = matriz_global[real][pred]
            fila += f"{val:>10}"
        print(fila)

    # Análisis de la matriz
    print(f"\n{'='*60}")
    print("ANÁLISIS DE LA MATRIZ DE CONFUSIÓN")
    print(f"{'='*60}")
    for real in clases:
        total_real = sum(matriz_global[real].values())
        correctos  = matriz_global[real][real]
        tasa       = correctos / total_real * 100 if total_real > 0 else 0
        errores    = [(p, matriz_global[real][p])
                      for p in clases if p != real and matriz_global[real][p] > 0]
        errores.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  {real}:")
        print(f"    Correctos    : {correctos}/{total_real} ({tasa:.1f}%)")
        if errores:
            print(f"    Se confunde con: ", end="")
            for clase_conf, cnt in errores[:3]:
                print(f"{clase_conf} ({cnt})", end="  ")
            print()
        else:
            print(f"    Sin confusiones significativas")

    return {
        "k"                  : k,
        "folds"              : resultados_folds,
        "accuracy_promedio"  : acc_prom,
        "accuracy_varianza"  : acc_var,
        "macro_f1_promedio"  : f1_prom,
        "macro_f1_varianza"  : f1_var,
        "metricas_promedio"  : metricas_promedio,
        "matriz_global"      : matriz_global,
        "clases"             : clases,
    }


# ============================================================
# BLOQUE PRINCIPAL
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("EVALUACIÓN DEL MODELO - K-FOLDS CROSS VALIDATION")
    print("=" * 60)

    # --- Cargar dataset limpio ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "processed", "tickets_clean.csv")

    if not os.path.exists(csv_path):
        print("ERROR: No se encontró tickets_clean.csv")
        print("Ejecuta primero: python3 data/prepare_dataset.py")
        exit(1)

    df = pd.read_csv(csv_path)
    print(f"\nDataset cargado: {len(df)} registros")
    print(f"Distribución:\n{df['category'].value_counts().to_string()}")

    texts  = df["text"].tolist()
    labels = df["category"].tolist()

    # --- Ejecutar K-Folds con K=5 ---
    resultados = evaluar_kfolds(texts, labels, k=5)

    print(f"\n{'='*60}")
    print("INTERPRETACIÓN PARA EL DOCUMENTO")
    print(f"{'='*60}")
    print(f"""
  Accuracy promedio  : {resultados['accuracy_promedio']*100:.2f}%
  Macro F1 promedio  : {resultados['macro_f1_promedio']:.4f}
  Varianza Accuracy  : {resultados['accuracy_varianza']:.6f}
  Varianza Macro F1  : {resultados['macro_f1_varianza']:.6f}

  Interpretación:
    - Una varianza cercana a 0 indica que el modelo es estable
      y generaliza bien en todos los folds.
    - Un Macro F1 alto indica buen rendimiento en TODAS las
      clases por igual, sin sesgarse hacia las más frecuentes.
    - La matriz de confusión muestra qué categorías se
      confunden entre sí y cuáles son más fáciles de predecir.
    """)

    print("\n✅ Etapa 4 completada.")
    print("   Puedes continuar con la Etapa 5: train.py")