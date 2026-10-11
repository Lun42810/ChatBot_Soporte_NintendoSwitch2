import os
import re
import groq
import streamlit as st
from dotenv import load_dotenv

# 1. Cargar variables de entorno obligatoriamente al inicio
load_dotenv()

# Importamos las funciones de tu archivo rag.py (permanecen intactas)
from rag import MAX_HISTORY_TURNS, RAGConfig, RAGError, answer_question, get_vector_store

# 2. Configuración estética de la página de Streamlit
st.set_page_config(
    page_title="ChatBot Soporte Nintendo Switch",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 Soporte Técnico Nintendo Switch")
st.write("Bienvenido al asistente de soporte RAG. Haz tus preguntas sobre la consola.")

# 3. Precarga del índice (Optimizado para Streamlit usando caché)
@st.cache_resource
def init_vector_store():
    try:
        get_vector_store(RAGConfig.from_env())
    except Exception as e:
        st.error(f"No se pudo precargar el índice: {e}")

init_vector_store()

# 4. Inicializar el historial de chat en la sesión de Streamlit si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Mostrar los mensajes anteriores del historial en la pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Capturar la nueva pregunta del usuario
if question := st.chat_input("¿En qué te puedo ayudar hoy?"):
    
    # Mostrar la pregunta del usuario inmediatamente en la interfaz
    with st.chat_message("user"):
        st.markdown(question)
    
    # Limitar la longitud del texto como hacías en Flask
    question_cleaned = question.strip()[:1000]

    # Preparar el historial en el formato que tu función answer_question() espera
    # Tomamos solo los últimos turnos configurados en tu RAG
    raw_history = st.session_state.messages[-MAX_HISTORY_TURNS:]
    history = [{"role": m["role"], "content": m["content"][:2000]} for m in raw_history]

    # Mostrar un spinner de carga mientras la IA responde
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # Llamada directa a tu lógica de RAG
                result = answer_question(question_cleaned, history)
                
                # Procesar la respuesta (quitar negritas como lo hacías con tu regex)
                clean_answer = re.sub(r"\*\*(.+?)\*\*", r"\1", result["answer"])
                
                # Mostrar la respuesta en la interfaz
                st.markdown(clean_answer)
                
                # Si deseas mostrar las fuentes de donde sacó la info (opcional pero muy útil)
                if result.get("sources"):
                    with st.expander("📚 Fuentes consultadas"):
                        for source in result["sources"]:
                            st.write(f"- {source}")

                # Guardar la interacción en el historial de la sesión para la próxima pregunta
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

            # Control de errores idéntico a tu Flask
            except RAGError as error:
                st.error(f"Error en el servicio: {str(error)}")
            except groq.RateLimitError:
                st.error("El servicio de IA alcanzó su límite de uso. Intenta de nuevo en unos minutos.")
            except Exception as e:
                st.error("No fue posible procesar la pregunta. Revisa la configuración y los registros.")
