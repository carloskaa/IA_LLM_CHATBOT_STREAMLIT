import streamlit as st
import whisper
import os
import tempfile
import subprocess
import base64
import sounddevice as sd
import wavio
from datetime import datetime
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import TFIDFRetriever
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

# ---------------- CONFIGURACIÓN ----------------
cohere_api_key = '1msKL9N3DxmNqmxMCQLQ4CHz8e1dO130v1urBoUI'
st.set_page_config(page_title="Chatbot Científico", layout="centered")

# ---------------- RUTA FFMPEG ----------------
# ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
# if os.path.isfile(ffmpeg_path):
#     def patched_run(cmd, **kwargs):
#         if cmd[0] == "ffmpeg":
#             cmd[0] = ffmpeg_path
#         return subprocess.run(cmd, **kwargs)
#     whisper.audio.run = patched_run

# ---------------- RUTA FFMPEG UNIVERSAL ----------------
def patched_run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

whisper.audio.run = patched_run  # Se usa ffmpeg del sistema

# ---------------- FONDO ----------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 2rem;
            background-color: transparent !important;
            color: white;
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: #FFFFFF !important;
        }}
        .stTextInput > div > div > input {{
            background-color: #1b1f2a;
            color: white;
        }}
        div.stButton > button:first-child {{
            background-color: black !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #555;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("fondo.jpeg")

# ---------------- TÍTULO Y LOGO ----------------
st.markdown("<h1 style='text-align: center; color: #A1F55A;'>Chatbot <span style='color:#8BF'>SCIENTIA</span></h1>", unsafe_allow_html=True)
with open("logo2.jpeg", "rb") as image_file:
    logo2_encoded = base64.b64encode(image_file.read()).decode()
st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{logo2_encoded}' width='100'/></div>", unsafe_allow_html=True)
# st.markdown("<h3 style='color: white; text-align: center;'>Bienvenido, Scientia es un chatbot diseñado para responder tus preguntas sobre el radar de la IA y temas afines. Fórmula tu pregunta abajo.</h3>", unsafe_allow_html=True)
st.markdown("""
<p style='color: white; text-align: center; font-size: 16px;'>
Bienvenido, <strong>Scientia</strong> es un chatbot diseñado para responder tus preguntas sobre el radar de la IA y temas afines. Fórmula tu pregunta abajo.
</p>
""", unsafe_allow_html=True)

# ---------------- CARGA DOCUMENTOS ----------------
@st.cache_resource
def load_documents():
    loader = PyPDFLoader("Guión_HMI.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

@st.cache_resource
def setup_qa_chain():
    texts = load_documents()
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key, user_agent="my-app")
    db = TFIDFRetriever.from_documents(texts)

    prompt_template = """
    ## Instructions
    You are an AI personal friendly assistant named Scientia, designed to answer questions about el radar de la IA.
    You MUST only support Spanish for questions and answers. Your responses should be concise and directly address the specific question.
    Answer based solely on the content of the provided documents. Do not generate an answer that is not supported by the documents.
    If you cannot find the answer to the user's question in the documents provided, respond by stating that the information is beyond your scope.

    Use the following documents and chat history to answer the question in Spanish:

    Question:{question}
    Documents: {context}
    Chat History: {history}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history"])
    llm = Cohere(model="command-r-08-2024", temperature=0.75, cohere_api_key=cohere_api_key, max_tokens=300)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db,
        chain_type_kwargs={
            "verbose": False,
            "prompt": PROMPT,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question")
        },
        verbose=False
    )
    return qa

qa_chain = setup_qa_chain()

# ---------------- SESIÓN ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- FUNCIONES DE AUDIO ----------------
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="Spanish")
    return result["text"]

def grabar_audio_y_guardar():
    duration = 10
    samplerate = 44100
    channels = 1
    filename = f"grabacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    st.info("🎙️ Grabando 10 segundos... habla ahora.")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
        sd.wait()
        wavio.write(filename, audio_data, samplerate, sampwidth=2)
        return filename
    except Exception as e:
        st.error(f"Error al grabar: {e}")
        return None

# ---------------- ENTRADAS ----------------
user_input = st.text_input("✍️ Escribe tu pregunta:", key="text_input")

col1, col2 = st.columns(2)

with col1:
    if st.button("Enviar pregunta escrita") and user_input:
        with st.spinner("🤖 Pensando..."):
            result = qa_chain.invoke({"query": user_input})
            st.session_state.chat_history.append(("Tú", user_input))
            st.session_state.chat_history.append(("Scientia", result["result"]))

with col2:
    if st.button("🔴 Grabar 10 segundos de voz"):
        audio_path = grabar_audio_y_guardar()
        if audio_path:
            st.audio(audio_path)
            with st.spinner("🎧 Transcribiendo grabación..."):
                transcription = transcribe_audio(audio_path)
                st.success("✅ Transcripción completada")
                st.markdown(f"<p style='color: white;'>📝 {transcription}</p>", unsafe_allow_html=True)
                final_input = transcription
                os.remove(audio_path)
            if final_input:
                with st.spinner("🤖 Pensando..."):
                    result = qa_chain.invoke({"query": final_input})
                    st.session_state.chat_history.append(("Tú", final_input))
                    st.session_state.chat_history.append(("Scientia", result["result"]))

# ---------------- HISTORIAL RECIENTE PRIMERO ----------------
st.markdown("---")
for speaker, text in reversed(st.session_state.chat_history):
    icon = "🧑" if speaker == "Tú" else "🤖"
    st.markdown(f"<p style='color: white;'><strong>{icon} {speaker}:</strong> {text}</p>", unsafe_allow_html=True)
