"""
=============================================================
ETAPA 2 - Módulo de Preprocesamiento de Texto
=============================================================

Este módulo implementa la función principal preprocess_text()
que transforma texto crudo en una lista de tokens limpios.

Pipeline aplicado:
    1. Conversión a minúsculas
    2. Eliminación de caracteres especiales y números
    3. Eliminación de signos de puntuación
    4. Tokenización (split por espacios)
    5. Eliminación de stopwords
    6. Stemming con PorterStemmer (NLTK)
    7. Eliminación de tokens vacíos o demasiado cortos

=============================================================
"""

import re
from nltk.stem import PorterStemmer

# -------------------------------------------------------
# INICIALIZACIÓN DEL STEMMER
# PorterStemmer no requiere datos descargados de NLTK,
# funciona con reglas morfológicas integradas en la librería.
# -------------------------------------------------------
_stemmer = PorterStemmer()

# -------------------------------------------------------
# STOPWORDS EN INGLÉS
# Lista curada manualmente con las palabras más frecuentes
# del idioma inglés y términos genéricos de soporte técnico
# que no aportan valor discriminativo al clasificador.
# -------------------------------------------------------
STOPWORDS = {
    # Pronombres
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    # Interrogativos / relativos
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "when", "where", "why", "how",
    # Verbos auxiliares
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "could", "should", "shall", "can", "may", "might",
    # Artículos y preposiciones
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there",
    # Cuantificadores
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very",
    # Contracciones y abreviaciones comunes
    "s", "t", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
    # Palabras genéricas de soporte (poco discriminativas)
    "get", "got", "getting", "also", "please", "need", "want",
    "issue", "problem", "help", "support", "customer", "service",
    "using", "use", "used", "time", "make", "made", "like",
    "know", "think", "look", "come", "go", "going", "still",
    "even", "back", "let", "us", "just", "now", "would",
    # Saludos y cortesías comunes en tickets
    "hi", "hello", "dear", "thank", "thanks", "regards",
    "sincerely", "im", "ive", "id", "ill",
}

# -------------------------------------------------------
# LONGITUD MÍNIMA DE TOKEN
# Tokens con menos de 3 caracteres generalmente no aportan
# valor semántico y aumentan el ruido en el vocabulario.
# -------------------------------------------------------
MIN_TOKEN_LENGTH = 3


def preprocess_text(text: str) -> list:
    """
    Transforma texto crudo en una lista de tokens procesados.

    Pipeline completo:
        1. Conversión a minúsculas
        2. Eliminación de URLs y correos
        3. Eliminación de números y caracteres especiales
        4. Eliminación de puntuación
        5. Tokenización por espacios
        6. Eliminación de stopwords
        7. Stemming con PorterStemmer
        8. Filtro de tokens vacíos o muy cortos

    Args:
        text (str): Texto crudo de entrada (subject + description).

    Returns:
        list: Lista de tokens limpios y con stemming aplicado.

    Ejemplo:
        >>> preprocess_text("I'm having issues with my billing invoice")
        ['hav', 'issu', 'bill', 'invoic']
    """

    # --- Paso 1: Validar entrada ---
    if not isinstance(text, str) or not text.strip():
        return []

    # --- Paso 2: Convertir a minúsculas ---
    text = text.lower()

    # --- Paso 3: Eliminar URLs (http://... o www...) ---
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # --- Paso 4: Eliminar correos electrónicos ---
    text = re.sub(r'\S+@\S+', ' ', text)

    # --- Paso 5: Eliminar números (solos o mezclados) ---
    text = re.sub(r'\b\d+\b', ' ', text)

    # --- Paso 6: Eliminar caracteres especiales y puntuación ---
    # Conservamos solo letras del alfabeto inglés y espacios
    text = re.sub(r'[^a-z\s]', ' ', text)

    # --- Paso 7: Eliminar espacios múltiples ---
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Paso 8: Tokenización (split por espacios) ---
    tokens = text.split()

    # --- Paso 9: Eliminar stopwords ---
    tokens = [t for t in tokens if t not in STOPWORDS]

    # --- Paso 10: Stemming con PorterStemmer ---
    # Reduce cada token a su raíz morfológica.
    # Ejemplo: "billing" -> "bill", "running" -> "run"
    tokens = [_stemmer.stem(t) for t in tokens]

    # --- Paso 11: Filtrar tokens muy cortos o vacíos ---
    tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]

    return tokens


def tokens_to_text(tokens: list) -> str:
    """
    Convierte una lista de tokens de vuelta a un string.
    Útil para inspección y debugging.

    Args:
        tokens (list): Lista de tokens procesados.

    Returns:
        str: Tokens unidos por espacios.
    """
    return " ".join(tokens)


# -------------------------------------------------------
# BLOQUE DE PRUEBA
# -------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("PRUEBA DEL MÓDULO DE PREPROCESAMIENTO")
    print("=" * 60)

    casos_prueba = [
        {
            "categoria": "Soporte Técnico",
            "texto": "Product setup I'm having an issue with the device. "
                     "I've tried troubleshooting steps but the problem persists."
        },
        {
            "categoria": "Facturación",
            "texto": "Billing error I was charged twice this month on my invoice. "
                     "Please check my account and issue a refund."
        },
        {
            "categoria": "Cancelación",
            "texto": "Account cancellation I want to cancel my subscription "
                     "and close my account permanently."
        },
        {
            "categoria": "Consulta General",
            "texto": "Product inquiry Can you tell me more about the features "
                     "available in the premium plan?"
        },
        {
            "categoria": "Queja",
            "texto": "Refund request I am very dissatisfied with the service. "
                     "The product stopped working after one week and I want my money back."
        },
    ]

    for caso in casos_prueba:
        tokens = preprocess_text(caso["texto"])
        print(f"\nCategoría : {caso['categoria']}")
        print(f"Original  : {caso['texto'][:80]}...")
        print(f"Tokens    : {tokens}")
        print(f"Total     : {len(tokens)} tokens")

    # -------------------------------------------------------
    # Prueba con el dataset real
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("PRUEBA CON DATASET REAL (primeras 5 filas)")
    print("=" * 60)

    import os, pandas as pd

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "processed", "tickets_clean.csv"
    )

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for i, row in df.head(5).iterrows():
            tokens = preprocess_text(row["text"])
            print(f"\n[{i}] Categoría: {row['category']}")
            print(f"    Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens)>10 else ''}")

        # Estadísticas generales sobre longitud de tokens
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE LONGITUD DE TOKENS")
        print("=" * 60)

        sample = df.sample(500, random_state=42)
        longitudes = [len(preprocess_text(t)) for t in sample["text"]]

        print(f"  Promedio de tokens por ticket : {sum(longitudes)/len(longitudes):.1f}")
        print(f"  Mínimo                        : {min(longitudes)}")
        print(f"  Máximo                        : {max(longitudes)}")
        print(f"  Tokens totales en muestra     : {sum(longitudes)}")
    else:
        print("  Dataset no encontrado. Ejecuta primero data/prepare_dataset.py")

    print("\nOK - Módulo preprocessing.py listo.")
    print("   Puedes continuar con la Etapa 3: naive_bayes.py")
