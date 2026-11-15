# 🎮 Chatbot RAG de Soporte Técnico - Nintendo Switch 2

**Repositorio:** https://github.com/Lun42810/ChatBot_Soporte_NintendoSwitch2  
**Rama de entrega:** `codex/avance3-tools`  
**Aplicación web:** https://nintendoavance3.streamlit.app/

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

Groq API para generación de respuestas. El modelo se configura con `GROQ_MODEL` (en la configuración local de esta entrega: `openai/gpt-oss-120b`).

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

git clone --branch codex/avance3-tools https://github.com/Lun42810/ChatBot_Soporte_NintendoSwitch2.git
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

Resultados de referencia de Avance 2: faithfulness 0.372, answer_relevancy 0.354, context_precision 0.617 y context_recall 0.442 (20 preguntas). En Avance 3 se realizó una muestra exploratoria de 5 preguntas con `qwen/qwen3.8-27b` como generador, `openai/gpt-oss-safeguard-20b` como juez y un solo fragmento recuperado por pregunta: faithfulness 0.527, answer_relevancy 0.619, context_precision 0.400 y context_recall 0.333. La muestra usa una configuración y un subconjunto diferentes, por lo que no demuestra una mejora frente a Avance 2.

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

## Avance 3: agente RAG con herramientas

La interfaz ahora llama a `agent.run_agent`, un bucle de tool calling con un máximo de 6 iteraciones por pregunta. Conserva la ingesta, Chroma y el recuperador de Avance 2.

El ciclo sigue el patrón del repositorio de clase [`tools_rags`](https://github.com/lpajarof/tools_rags): el modelo selecciona una tool con `bind_tools`, el programa la ejecuta con `.invoke()` y devuelve el resultado en un `ToolMessage`. Las funciones se registran con `StructuredTool.from_function`. El ejemplo web del repositorio de clase usa Tavily y requiere `TAVILY_API_KEY`; este proyecto consulta directamente el catálogo oficial de Nintendo para no depender de esa clave.

### Arquitectura y herramientas

El agente expone cinco tools registradas con `StructuredTool.from_function`; las herramientas que requieren parámetros usan esquemas tipados. El bucle de `agent.py` permite al LLM seleccionar y encadenar herramientas, mientras la aplicación ejecuta cada llamada mediante `.invoke()` y devuelve el resultado al modelo.

- `buscar_en_base(consulta)`: primera herramienta para preguntas sobre los manuales. Devuelve como máximo 4 fragmentos de 350 caracteres, con archivo, página y fragmento. El índice y los documentos permanecen locales; solo los resultados limitados se envían al LLM de Groq para responder.
- `convertir_unidades(valor, desde, hacia)`: tarea determinista que convierte mm, cm, m, pulgadas, pies, °C y °F. El modelo extrae sus argumentos de la pregunta.
- `buscar_juegos_nintendo(consulta)`: consulta el catálogo público oficial de Nintendo para títulos y datos disponibles; úsala después de revisar la base cuando pregunten por juegos, géneros, jugadores, modos o recomendaciones. Devuelve como máximo 3 resultados.
- `comparar_juegos_nintendo(juego_a, juego_b)`: tarea de dominio que busca cada título en el mismo catálogo y prepara dos fichas comparables. Devuelve hasta 2 fichas con sus URLs oficiales. No estima precio ni características ausentes en la fuente.
- `buscar_web(consulta)`: usa la misma fuente oficial, fija en el código en `nintendo.com`; no recibe fragmentos del corpus. La página del catálogo se descarga directamente con HTTP y sus datos estructurados se analizan localmente, sin proveedor de búsqueda ni clave adicional. La copia se mantiene en memoria durante una hora y cada llamada devuelve como máximo 3 resultados.

El saludo y la charla casual se responden sin llamar al LLM ni tools. El prompt está en `agent.py`: ordena buscar primero en la base, define cuándo usar cada tool, trata el material recuperado como datos no confiables, mantiene las citas separadas (`[Fuente: documento, pág. N]` y `[web: URL]`) y conserva la respuesta exacta `No encontré información sobre esto en la base de conocimientos.` cuando no hay evidencia. El panel desplegable de cada respuesta muestra herramientas y argumentos, y presenta las fuentes por origen.

Cada turno se registra como una línea JSON en `logs/agent.jsonl` (ruta configurable con `AGENT_LOG_PATH`), incluyendo nombres de tools, argumentos, cantidad de llamadas al LLM, tokens disponibles en metadatos y duración. Para evitar escribir en un disco efímero durante despliegues, define `AGENT_LOG_PATH` en un volumen persistente o usa el almacenamiento de logs de la plataforma.

### Instalación y ejecución

Instala dependencias con `pip install -r requirements.txt`, configura `GROQ_API_KEY` y `GROQ_MODEL` en `.env` o en los secrets del proveedor. La búsqueda del catálogo no requiere clave adicional, solo conexión a internet desde el servidor de Streamlit. No guardes claves en Git. Indexa los manuales con `python ingest.py --rebuild` y ejecuta `streamlit run app.py`.

Para demostrar la tool de base sin llamar al LLM: `python tool_demo.py "¿Cómo se conecta la cámara?"`. La evaluación está en `evaluation/run_agent_eval.py`; `evaluation/behavior_cases.json` declara nueve comportamientos esperados antes de su ejecución. Ejecuta `python evaluation/run_agent_eval.py --ids 1-15 --run-behavior-cases` para correr las mismas preguntas con el pipeline fijo y con el agente, medir Ragas para ambos, guardar tools/argumentos y comparar promedios de llamadas, tokens y tiempo en `evaluation/results/agent_eval.json`. El baseline histórico completo de Avance 2 está en `evaluation/results/iteracion_chunk300_resumen.json` (20 preguntas): faithfulness 0.372, answer_relevancy 0.354, context_precision 0.617 y context_recall 0.442. Para una comparación directa debe usarse la evaluación pareada nueva con el mismo subconjunto de IDs.

### Despliegue

La aplicación de esta entrega está desplegada en Streamlit Community Cloud: https://nintendoavance3.streamlit.app/. Configura `GROQ_API_KEY` (y opcionalmente `GROQ_MODEL`) en **Settings → Secrets** de Streamlit Cloud; no guardes claves en Git. El despliegue de la entrega usa la rama `codex/avance3-tools` del repositorio enlazado arriba.
