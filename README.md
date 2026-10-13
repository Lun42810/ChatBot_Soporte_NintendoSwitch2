# 🎮 Chatbot RAG de Soporte Técnico - Nintendo Switch 2

Aplicación web interactiva desarrollada con **Streamlit** y un diseño retro de consola, capaz de responder preguntas sobre manuales oficiales de accesorios y componentes de Nintendo Switch 2. Utiliza un pipeline de Recuperación Aumentada por Generación (RAG): busca fragmentos pertinentes en los documentos PDF mediante similitud semántica y entrega dicho contexto a un modelo de lenguaje en Groq para redactar una respuesta precisa con referencias a la fuente y la página.

---

## 📁 Contenido del proyecto

```text
.
├── app.py                  # Interfaz de usuario y lógica de chat en Streamlit
├── rag.py                  # Pipeline RAG: carga, chunking, índice ChromaDB y generación
├── ingest.py               # Script de ingesta para construir/actualizar el índice vectorial
├── requirements.txt        # Dependencias actualizadas del proyecto
├── .env.example            # Plantilla de variables de entorno
├── .gitignore              # Exclusión de secretos y datos generados
├── documents/              # Manuales PDF oficiales consultados
├── evaluation/             # Evaluación formal con el framework Ragas
│   ├── eval_questions.json # Dataset de evaluación (preguntas corpus y fuera de alcance)
│   ├── run_ragas.py        # Script de ejecución de métricas Ragas
│   └── compare_runs.py     # Script de comparación de experimentos (baseline vs iter1)
└── chroma/                 # Índice vectorial persistente local (no se versiona)

Los PDF en documents/ incluyen manuales de la consola (información importante), adaptador de corriente, cámara, funda y protector de pantalla, base, mandos Joy-Con 2, empuñadura de carga, volante, mando Pro y mando de Nintendo GameCube.

🛠️ Tecnologías y Stack
Streamlit para la interfaz web conversacional y la gestión nativa del estado del chat.

Python como lenguaje base del motor de inferencia y procesamiento.

LangChain para la gestión de documentos, división de texto (splitting), conectividad con bases vectoriales e integración con Groq.

PyPDF para la extracción de texto estructurado desde los manuales PDF.

Sentence Transformers (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) para la generación local de embeddings en CPU.

ChromaDB como base de datos vectorial para el almacenamiento persistente y búsqueda por distancia coseno.

Groq API con el modelo qwen/qwen3.8-27b para la generación de respuestas en lenguaje natural.

⚙️ Flujo RAG e Implementación Técnica
Selección de documentos: Corpus de 10 manuales oficiales en PDF, procesados en 1,602 fragmentos (chunks).

Ingesta y Extracción: Extracción de texto y metadatos (source, page) utilizando PyPDFLoader.

Chunking: Fragmentación recursiva (RecursiveCharacterTextSplitter) configurada con un tamaño de 500 caracteres y un solape de 50 caracteres (10%).

Vectorización y Almacenamiento: Embeddings multilingües almacenados en la base de datos vectorial local ChromaDB.

Recuperación: Búsqueda por similitud semántica con un top_k = 5 y filtrado optimizado para el idioma español.

Generación: Integración con Groq (qwen/qwen3.8-27b) aplicando directrices estrictas de restricciones, formato de citas y control de alucinaciones.

💻 Instalación y Ejecución Local
Requisitos previos
Python 3.10 o posterior.

Git.

Una API Key activa de la Groq Console.

Pasos de Instalación
1.Clonar el repositorio:

git clone <URL_DEL_REPOSITORIO>
cd ChatBot_Soporte_NintendoSwitch2

2.Crear y activar un entorno virtual:

En Windows (PowerShell):
py -m venv venv
.\venv\Scripts\Activate.ps1

En macOS / Linux:
python3 -m venv venv
source venv/bin/activate

3.Instalar las dependencias:
python -m pip install --upgrade pip
pip install -r requirements.txt

4.Configurar las variables de entorno:
Copia el archivo de ejemplo a un nuevo archivo .env:

En PowerShell: Copy-Item .env.example .env

En macOS/Linux: cp .env.example .env

Abre el archivo .env e introduce tus credenciales:

GROQ_API_KEY=tu_clave_real_de_groq
GROQ_MODEL=qwen/qwen3.8-27b
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
LANGUAGES=es

5.Indexar los documentos y ejecutar la aplicación:
python ingest.py --rebuild
streamlit run app.py

Se abrirá automáticamente tu navegador en http://localhost:8501 con la interfaz gráfica de Streamlit.

📊 Evaluación RAG con Ragas
El proyecto incluye un módulo de evaluación cuantitativa formal mediante el framework Ragas y un LLM evaluador (openai/gpt-oss-120b).

Dataset de Evaluación (evaluation/eval_questions.json): Compuesto por 20 preguntas de prueba (15 dentro del corpus de manuales y 5 casos fuera de alcance para medir control de abstención).

Iteración y Comparativa: Se evaluó el impacto de activar filtros estrictos de idioma, logrando reducir abstenciones falsas y mejorando sustancialmente las métricas de relevancia de respuesta (answer_relevancy).

Comandos de ejecución de pruebas:
# Ejecutar evaluación con Ragas
python evaluation/run_ragas.py --label iter1 --languages es

# Comparar ejecuciones previas
python evaluation/compare_runs.py baseline iter1

☁️ Despliegue en la Nube
Para desplegar la aplicación web basada en Streamlit en plataformas como Streamlit Community Cloud o Render:

Build Command: pip install -r requirements.txt

Start Command (en Streamlit / Render): streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

Asegúrate de configurar la variable de entorno secreta GROQ_API_KEY en el panel de control de tu servicio en la nube.