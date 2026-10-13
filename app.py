import os
import re
import groq
from dotenv import load_dotenv
import streamlit as st

# Cargar configuración del .env
load_dotenv()

from rag import MAX_HISTORY_TURNS, RAGConfig, RAGError, answer_question, get_vector_store

# 1. Configuración de la página de Streamlit
st.set_page_config(
    page_title="Nintendo RPG Chatbot",
    page_icon="🎮",
    layout="centered"
)

# 2. Inyectar el CSS retro
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .stApp {
            background-color: #1a1c23;
            background-image: radial-gradient(#2c303b 15%, transparent 16%);
            background-size: 16px 16px;
            font-family: 'Press Start 2P', monospace;
            color: #ffffff;
        }

        .game-header {
            background-color: #e60012;
            padding: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 4px solid #000;
            margin-bottom: 20px;
        }

        .game-title {
            font-size: 10px;
            color: #fff;
            text-shadow: 2px 2px #000;
            letter-spacing: 1px;
        }

        .stChatMessage {
            font-family: 'Press Start 2P', monospace !important;
            font-size: 9px !important;
            border-radius: 0px !important;
        }

        div.stButton > button {
            background-color: #ffe600;
            color: #000;
            border: 3px solid #000;
            font-family: 'Press Start 2P', monospace;
            font-size: 8px;
            padding: 8px 12px;
            cursor: pointer;
            box-shadow: 2px 2px 0px #fff;
            border-radius: 0px;
        }
        div.stButton > button:active {
            transform: translate(2px, 2px);
            box-shadow: 0px 0px 0px #fff;
        }

        .source-item {
            margin-top: 6px;
            padding: 8px;
            background: #000;
            border: 2px solid #415a77;
            color: #e0e1dd;
            font-size: 8px;
        }
        .source-item b {
            display: block;
            color: #39e75f;
            margin-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Precarga del índice
@st.cache_resource
def inicializar_rag():
    try:
        return get_vector_store(RAGConfig.from_env())
    except Exception as e:
        st.error(f"No se pudo precargar el índice: {e}")
        return None

inicializar_rag()

# 4. Función para limpiar historial
def _clean_history(raw):
    """Valida el historial y conserva solo los últimos turnos."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("'history' debe ser una lista.")
    cleaned = []
    for item in raw[-MAX_HISTORY_TURNS:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            cleaned.append({"role": item["role"], "content": item["content"][:2000]})
    return cleaned

# Cabecera visual simulada de la consola
st.markdown("""
    <div class="game-header">
        <span class="game-title">🎮 NINTENDO CHAT v1.0</span>
    </div>
""", unsafe_allow_html=True)

WELCOME = "¡Hola, Jugador 1! Pregúntame sobre los manuales de Nintendo Switch 2. Puedes hacer preguntas de seguimiento."

# 5. Gestión del estado de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME, "sources": []}
    ]

# Botón de nueva partida
if st.sidebar.button("NUEVA PARTIDA"):
    st.session_state.messages = [{"role": "assistant", "content": WELCOME, "sources": []}]
    st.rerun()

# Renderizar historial
for message in st.session_state.messages:
    avatar = "🕹️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"FUENTES ({len(message['sources'])})"):
                for src in message["sources"]:
                    source_name = src.get('source', 'Desconocido')
                    page_num = src.get('page', '?')
                    snippet_text = src.get('snippet', '')
                    st.markdown(f"""
                        <div class="source-item">
                            <b>{source_name} - pág. {page_num}</b>
                            <span>{snippet_text[:350]}...</span>
                        </div>
                    """, unsafe_allow_html=True)

# 6. Entrada del usuario
if user_query := st.chat_input("Escribe un comando..."):
    question = user_query.strip()[:1000]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # Procesamiento del bot
    with st.chat_message("assistant", avatar="🕹️"):
        with st.spinner("Buscando en los manuales..."):
            try:
                history_raw = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                history = _clean_history(history_raw)

                # 🔹 Llamada única al pipeline RAG
                result = answer_question(question, history)

                #  Mostrar cómo el bot reformuló la pregunta de seguimiento
                st.caption(f"🔍 Interpretación interna: {result['standalone_question']}")

                response_text = re.sub(r"\*\*(.+?)\*\*", r"\1", result["answer"])
                sources = result["sources"]

                # Agregar las fuentes al texto para que queden en el historial
                if sources:
                    fuentes_unicas = list({src.get('source', 'Desconocido') for src in sources})
                    fuentes_texto = "\nFuentes: " + "; ".join(
                        f"{nombre} (pág. {src.get('page', '?')})"
                        for nombre in fuentes_unicas
                        for src in sources if src.get('source') == nombre
                    )
                    response_text += "\n" + fuentes_texto

                st.markdown(response_text)

                # 🔹 Mostrar fuentes en bloque expandible
                if sources:
                    with st.expander(f"FUENTES ({len(sources)})"):
                        for src in sources:
                            st.markdown(f"""
                                <div class="source-item">
                                    <b>{src.get('source', 'Desconocido')} - pág. {src.get('page', '?')}</b>
                                    <span>{src.get('snippet', '')[:350]}...</span>
                                </div>
                            """, unsafe_allow_html=True)

                # Guardar respuesta y fuentes en el estado
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })

            except RAGError as error:
                error_msg = str(error)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
                st.rerun()
            except groq.RateLimitError as error:
                print("RATE LIMIT:", str(error)[:400])
                error_msg = "El servicio de IA alcanzó su límite de uso. Intenta de nuevo en unos minutos."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
                st.rerun()
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"ERROR: {type(e).__name__}: {str(e)[:300]}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
                st.rerun()
