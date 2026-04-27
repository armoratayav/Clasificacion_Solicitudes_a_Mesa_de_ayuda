"""
=============================================================
ETAPA 3 - Implementación Manual de Naïve Bayes Multinomial
=============================================================

Fundamento matemático:
-----------------------
Dado un documento d con palabras w1, w2, ..., wn,
queremos encontrar la clase c* que maximiza:

    c* = argmax P(c) * PROD P(wi | c)
              c

Aplicando logaritmos para evitar underflow numérico:

    c* = argmax log P(c) + SUM log P(wi | c)
              c

Probabilidad a priori:
    P(c) = documentos_de_clase_c / total_documentos

Probabilidad condicional con Laplace Smoothing:
    P(w | c) = (count(w, c) + 1) / (total_words(c) + |V|)

Donde:
    count(w, c)    = veces que la palabra w aparece en clase c
    total_words(c) = total de palabras en clase c
    |V|            = tamaño del vocabulario
=============================================================
"""

import math
import pickle
import json
import os
from collections import defaultdict

# Importamos solo nuestro módulo de preprocesamiento
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.preprocessing import preprocess_text


class MultinomialNaiveBayes:
    """
    Clasificador Naïve Bayes Multinomial implementado desde cero.

    Atributos internos del modelo entrenado:
        classes         : lista de clases únicas
        vocabulary      : conjunto de todas las palabras vistas
        class_doc_counts: cantidad de documentos por clase
        class_priors    : log(P(c)) por clase
        word_counts     : frecuencia de cada palabra por clase
        total_words     : total de palabras por clase
    """

    def __init__(self):
        self.classes          = []       # ['Soporte Técnico', 'Facturación', ...]
        self.vocabulary       = set()    # Vocabulario global (Bag of Words)
        self.class_doc_counts = {}       # {clase: num_documentos}
        self.class_priors     = {}       # {clase: log P(clase)}
        self.word_counts      = {}       # {clase: {palabra: conteo}}
        self.total_words      = {}       # {clase: total palabras}
        self._trained         = False

    # -----------------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------------
    def fit(self, texts: list, labels: list):
        """
        Entrena el modelo con una lista de textos y sus etiquetas.

        Pasos internos:
            1. Preprocesar cada texto -> tokens
            2. Construir vocabulario global (Bag of Words)
            3. Contar palabras por clase
            4. Calcular probabilidades a priori log P(c)
            5. Almacenar conteos para calcular P(w|c) en inferencia

        Args:
            texts  (list): Lista de strings de texto crudo.
            labels (list): Lista de etiquetas correspondientes.
        """
        if len(texts) != len(labels):
            raise ValueError("texts y labels deben tener la misma longitud.")

        total_docs = len(texts)

        # --- Paso 1: Identificar clases únicas ---
        self.classes = sorted(list(set(labels)))

        # --- Paso 2: Inicializar estructuras ---
        self.class_doc_counts = {c: 0 for c in self.classes}
        self.word_counts      = {c: defaultdict(int) for c in self.classes}
        self.total_words      = {c: 0 for c in self.classes}
        self.vocabulary       = set()

        # --- Paso 3: Procesar cada documento ---
        for text, label in zip(texts, labels):
            tokens = preprocess_text(text)

            # Contar documento en su clase
            self.class_doc_counts[label] += 1

            # Contar cada token en la clase correspondiente
            for token in tokens:
                self.word_counts[label][token] += 1
                self.total_words[label]        += 1
                self.vocabulary.add(token)

        # --- Paso 4: Calcular probabilidades a priori ---
        # Usamos logaritmo para evitar underflow numérico
        # log P(c) = log(docs_en_clase / total_docs)
        self.class_priors = {}
        for c in self.classes:
            self.class_priors[c] = math.log(
                self.class_doc_counts[c] / total_docs
            )

        self._trained = True

        # --- Resumen del entrenamiento ---
        print(f"  Modelo entrenado con {total_docs} documentos")
        print(f"  Clases    : {self.classes}")
        print(f"  Vocabulario: {len(self.vocabulary)} palabras únicas")
        for c in self.classes:
            print(f"  {c:<22}: {self.class_doc_counts[c]} docs, "
                  f"{self.total_words[c]} palabras")

    # -----------------------------------------------------------
    # PROBABILIDAD CONDICIONAL CON LAPLACE SMOOTHING
    # -----------------------------------------------------------
    def _log_likelihood(self, token: str, clase: str) -> float:
        """
        Calcula log P(token | clase) con Laplace Smoothing.

        Fórmula:
            P(w | c) = (count(w, c) + 1) / (total_words(c) + |V|)

        El +1 en el numerador y +|V| en el denominador garantizan
        que ninguna probabilidad sea 0, incluso para palabras
        que no se vieron durante el entrenamiento.

        Args:
            token (str): Palabra a evaluar.
            clase (str): Clase para la que calcular la probabilidad.

        Returns:
            float: log P(token | clase)
        """
        vocab_size   = len(self.vocabulary)
        count_word   = self.word_counts[clase].get(token, 0)
        total        = self.total_words[clase]

        # Laplace Smoothing
        probability  = (count_word + 1) / (total + vocab_size)

        return math.log(probability)

    # -----------------------------------------------------------
    # INFERENCIA - SCORE POR CLASE
    # -----------------------------------------------------------
    def _compute_scores(self, text: str) -> dict:
        """
        Calcula el score logarítmico para cada clase.

        Fórmula:
            score(c) = log P(c) + SUM log P(wi | c)
                                   wi in tokens

        Args:
            text (str): Texto crudo de entrada.

        Returns:
            dict: {clase: score_logaritmico}
        """
        if not self._trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")

        tokens = preprocess_text(text)

        scores = {}
        for clase in self.classes:
            # Iniciar con el prior logarítmico
            score = self.class_priors[clase]

            # Sumar log-verosimilitudes de cada token
            for token in tokens:
                score += self._log_likelihood(token, clase)

            scores[clase] = score

        return scores

    # -----------------------------------------------------------
    # PREDICCIÓN - CLASE MÁS PROBABLE
    # -----------------------------------------------------------
    def predict(self, text: str) -> str:
        """
        Predice la clase con mayor score logarítmico.

        Args:
            text (str): Texto crudo de entrada.

        Returns:
            str: Nombre de la clase predicha.
        """
        scores = self._compute_scores(text)
        return max(scores, key=scores.get)

    # -----------------------------------------------------------
    # PREDICCIÓN - PROBABILIDADES NORMALIZADAS POR CLASE
    # -----------------------------------------------------------
    def predict_proba(self, text: str) -> dict:
        """
        Calcula probabilidades normalizadas para cada clase.

        Los scores logarítmicos se convierten a probabilidades
        usando softmax para que sumen 1 y sean interpretables
        como porcentajes de confianza.

        Softmax:
            P(c) = exp(score_c) / SUM exp(score_i)

        Se aplica estabilización numérica restando el máximo score
        antes de calcular exponenciales (evita overflow).

        Args:
            text (str): Texto crudo de entrada.

        Returns:
            dict: {clase: probabilidad} donde sum(probabilidades) = 1
        """
        scores = self._compute_scores(text)

        # Estabilización numérica: restar el máximo
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total_exp  = sum(exp_scores.values())

        # Normalizar
        probas = {c: round(v / total_exp, 4) for c, v in exp_scores.items()}

        # Ordenar de mayor a menor probabilidad
        probas = dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))

        return probas

    # -----------------------------------------------------------
    # PERSISTENCIA - GUARDAR MODELO
    # -----------------------------------------------------------
    def save(self, path: str):
        """
        Guarda el modelo entrenado en un archivo pickle.

        El archivo contiene todo lo necesario para hacer
        predicciones sin necesidad de reentrenar:
            - classes
            - vocabulary
            - class_doc_counts
            - class_priors
            - word_counts
            - total_words

        Args:
            path (str): Ruta donde guardar el archivo .pkl
        """
        if not self._trained:
            raise RuntimeError("No hay modelo entrenado para guardar.")

        model_data = {
            "classes"         : self.classes,
            "vocabulary"      : list(self.vocabulary),
            "class_doc_counts": self.class_doc_counts,
            "class_priors"    : self.class_priors,
            "word_counts"     : {c: dict(v) for c, v in self.word_counts.items()},
            "total_words"     : self.total_words,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        size_kb = os.path.getsize(path) / 1024
        print(f"  Modelo guardado en: {path} ({size_kb:.1f} KB)")

    # -----------------------------------------------------------
    # PERSISTENCIA - CARGAR MODELO
    # -----------------------------------------------------------
    @staticmethod
    def load(path: str) -> "MultinomialNaiveBayes":
        """
        Carga un modelo previamente guardado desde un archivo pickle.

        Args:
            path (str): Ruta del archivo .pkl a cargar.

        Returns:
            MultinomialNaiveBayes: Instancia del modelo cargado y listo.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el modelo en: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = MultinomialNaiveBayes()
        model.classes          = data["classes"]
        model.vocabulary       = set(data["vocabulary"])
        model.class_doc_counts = data["class_doc_counts"]
        model.class_priors     = data["class_priors"]
        model.word_counts      = {c: defaultdict(int, v)
                                   for c, v in data["word_counts"].items()}
        model.total_words      = data["total_words"]
        model._trained         = True

        print(f"  Modelo cargado desde: {path}")
        print(f"  Clases: {model.classes}")
        print(f"  Vocabulario: {len(model.vocabulary)} palabras")

        return model

    def __repr__(self):
        if self._trained:
            return (f"MultinomialNaiveBayes("
                    f"clases={len(self.classes)}, "
                    f"vocabulario={len(self.vocabulary)})")
        return "MultinomialNaiveBayes(sin entrenar)"


# -------------------------------------------------------
# BLOQUE DE PRUEBA
# Ejecutar directamente: python3 naive_bayes.py
# -------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("PRUEBA DEL MODELO NAÏVE BAYES")
    print("=" * 60)

    # --- Cargar dataset ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "processed", "tickets_clean.csv")

    if not os.path.exists(csv_path):
        print("ERROR: No se encontró tickets_clean.csv")
        print("Ejecuta primero: python3 data/prepare_dataset.py")
        exit(1)

    df = pd.read_csv(csv_path)
    print(f"\nDataset cargado: {len(df)} registros")

    # --- Entrenamiento con 80% de los datos ---
    print("\n" + "=" * 60)
    print("ENTRENANDO con 80% del dataset...")
    print("=" * 60)

    split = int(len(df) * 0.8)
    df    = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_texts  = df["text"].iloc[:split].tolist()
    train_labels = df["category"].iloc[:split].tolist()
    test_texts   = df["text"].iloc[split:].tolist()
    test_labels  = df["category"].iloc[split:].tolist()

    model = MultinomialNaiveBayes()
    model.fit(train_texts, train_labels)

    # --- Evaluación rápida en el conjunto de prueba ---
    print("\n" + "=" * 60)
    print("EVALUACIÓN RÁPIDA (20% test)")
    print("=" * 60)

    correctos = 0
    for text, true_label in zip(test_texts, test_labels):
        pred = model.predict(text)
        if pred == true_label:
            correctos += 1

    accuracy = correctos / len(test_texts)
    print(f"  Accuracy rápida: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Correctos: {correctos} / {len(test_texts)}")

    # --- Ejemplos de predicción ---
    print("\n" + "=" * 60)
    print("EJEMPLOS DE PREDICCIÓN")
    print("=" * 60)

    ejemplos = [
        ("Billing invoice charged twice this month on my account",
         "Facturación"),
        ("My internet connection keeps dropping and I cannot connect",
         "Soporte Técnico"),
        ("I want to cancel my subscription and close my account",
         "Cancelación"),
        ("What features are included in the premium plan",
         "Consulta General"),
        ("Very disappointed with the quality of service I received",
         "Queja"),
    ]

    for texto, esperado in ejemplos:
        prediccion = model.predict(texto)
        probas     = model.predict_proba(texto)
        correcto   = "✅" if prediccion == esperado else "❌"

        print(f"\n  Texto     : {texto}")
        print(f"  Esperado  : {esperado}")
        print(f"  Predicho  : {prediccion} {correcto}")
        print(f"  Confianza : {probas[prediccion]*100:.1f}%")
        print(f"  Scores    : ", end="")
        for c, p in probas.items():
            print(f"{c[:8]}={p:.2f}", end="  ")
        print()

    # --- Guardar modelo de prueba ---
    print("\n" + "=" * 60)
    print("GUARDANDO Y CARGANDO MODELO (prueba)")
    print("=" * 60)

    test_pkl = os.path.join(base_dir, "model_test.pkl")
    model.save(test_pkl)

    model2 = MultinomialNaiveBayes.load(test_pkl)
    pred2  = model2.predict("I was charged twice on my billing account")
    print(f"  Predicción con modelo recargado: {pred2}")

    # Limpiar archivo de prueba
    if os.path.exists(test_pkl):
        os.remove(test_pkl)
        print("  Archivo de prueba eliminado.")

    print("\n✅ naive_bayes.py listo.")
    print("   Puedes continuar con la Etapa 4: evaluate.py")