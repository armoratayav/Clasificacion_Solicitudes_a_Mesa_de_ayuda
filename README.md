# Clasificacion de Solicitudes a Mesa de Ayuda

Sistema de clasificacion automatica de solicitudes de mesa de ayuda usando el dataset Bitext Customer Support LLM Chatbot Training Dataset. El modelo es un clasificador Multinomial Naive Bayes implementado manualmente, sin scikit-learn, Hugging Face `datasets` ni librerias que entrenen Naive Bayes automaticamente.

## Tecnologias

- Python 3.11
- Pandas para lectura y preparacion del CSV
- NLTK para stemming con PorterStemmer
- Flask y Flask-CORS para la API REST
- HTML, CSS y JavaScript para el frontend
- Pickle para guardar y cargar el modelo entrenado

## Instalacion

1. Crear y activar el entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Colocar el dataset Bitext en:

```text
data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

En este workspace tambien se soporta el nombre alternativo:

```text
data/raw/bitext_customer_support.csv
```

## Dataset

Dataset usado:

```text
Bitext Customer Support LLM Chatbot Training Dataset
```

Columnas originales esperadas:

```text
flags
instruction
category
intent
response
```

Columnas usadas por el modelo:

```text
instruction -> text
category    -> category
```

La columna `response` no se usa para entrenar porque representa la respuesta del chatbot, no la solicitud del cliente.

Categorias detectadas:

```text
ACCOUNT
CANCEL
CONTACT
DELIVERY
FEEDBACK
INVOICE
ORDER
PAYMENT
REFUND
SHIPPING
SUBSCRIPTION
```

## Uso

Ejecutar el flujo completo en este orden:

```powershell
.\.venv\Scripts\python.exe backend\prepare_dataset.py
.\.venv\Scripts\python.exe backend\evaluate.py
.\.venv\Scripts\python.exe backend\train.py
.\.venv\Scripts\python.exe backend\app.py
```

Scripts opcionales de prueba:

```powershell
.\.venv\Scripts\python.exe backend\preprocessing.py
.\.venv\Scripts\python.exe backend\naive_bayes.py
```

La API queda disponible en:

```text
http://localhost:5000
```

Endpoints:

- `GET /`: informacion general del servidor
- `GET /health`: estado del modelo cargado
- `POST /predict`: clasifica un ticket

Ejemplo de request:

```json
{
  "subject": "Order status",
  "description": "I want to check the status of my order"
}
```

Ejemplo de response:

```json
{
  "ticket_id": "TCK-1906259675",
  "category": "ORDER",
  "scores": {
    "ACCOUNT": 0.01,
    "CANCEL": 0.0,
    "ORDER": 0.95,
    "REFUND": 0.01
  }
}
```

Para usar el frontend, mantener Flask activo y abrir:

```text
frontend/index.html
```

## Arquitectura

```text
data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
        |
        v
backend/prepare_dataset.py
        |
        v
data/processed/tickets_clean.csv
        |
        v
backend/preprocessing.py
        |
        v
backend/naive_bayes.py
        |
        +--> backend/evaluate.py  -> K-Folds, metricas y matriz de confusion
        |
        +--> backend/train.py     -> backend/model.pkl
                                      |
                                      v
backend/app.py  <---------------- frontend/script.js
```

## Descripcion de archivos

### Backend

- `backend/prepare_dataset.py`: carga el CSV Bitext, valida `instruction` y `category`, elimina placeholders, limpia registros invalidos, normaliza categorias y genera `data/processed/tickets_clean.csv`.
- `backend/preprocessing.py`: convierte texto crudo en tokens mediante limpieza de placeholders, minusculas, eliminacion de URLs, correos, numeros, caracteres especiales, stopwords, stemming y tokens cortos.
- `backend/naive_bayes.py`: implementa manualmente Multinomial Naive Bayes con Bag of Words, conteos por clase, probabilidades a priori, Laplace smoothing, inferencia con logaritmos, `predict`, `predict_proba`, `save` y `load`.
- `backend/evaluate.py`: implementa K-Folds manual con K=5, accuracy, precision, recall, F1, Macro F1, varianza y matriz de confusion dinamica.
- `backend/train.py`: entrena el modelo final con el 100% del dataset limpio, guarda `backend/model.pkl` y verifica predicciones de ejemplo.
- `backend/app.py`: API Flask que carga `backend/model.pkl` y expone endpoints REST.
- `backend/model.pkl`: modelo entrenado y serializado.

### Datos

- `data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`: dataset Bitext esperado por el enunciado.
- `data/raw/bitext_customer_support.csv`: nombre alternativo soportado en este workspace.
- `data/processed/tickets_clean.csv`: dataset limpio con columnas exactas `text,category`.

### Frontend

- `frontend/index.html`: interfaz del formulario y panel de resultados.
- `frontend/styles.css`: estilos visuales.
- `frontend/script.js`: validacion del formulario, llamada a `POST /predict` y render dinamico de barras con las clases recibidas en `scores`.

## Modelo y resultados

Datos generados con el procesamiento actual:

- Filas originales del CSV: 26,872
- Registros limpios de entrenamiento: 24,266
- Categorias detectadas: 11
- Vocabulario final: 2,426 palabras
- Accuracy promedio en K-Folds: 99.20%
- Macro F1 promedio: 0.9914

Notas:

- El modelo no usa `response`.
- Las categorias se conservan en ingles y en mayusculas, tal como vienen en Bitext.
- No se usa `sklearn.naive_bayes.MultinomialNB`, `CountVectorizer`, `TfidfVectorizer` ni metricas de scikit-learn.
- No se usa `from datasets import load_dataset`; el proyecto trabaja directamente con el CSV local.
