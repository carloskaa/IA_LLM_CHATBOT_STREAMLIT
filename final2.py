import streamlit as st
import whisper
import os
import tempfile
import subprocess
import base64
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

# ---------------- RUTA MANUAL A FFMPEG ----------------
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
if os.path.isfile(ffmpeg_path):
    def patched_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            cmd[0] = ffmpeg_path
        return subprocess.run(cmd, **kwargs)
    whisper.audio.run = patched_run

# ---------------- FONDO Y ESTILO ----------------
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
        section[data-testid="stFileUploader"] > div {{
            background-color: #1b1f2a !important;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 0.5rem;
            color: white !important;
            max-width: 300px;
        }}
        section[data-testid="stFileUploader"] button {{
            background-color: #000 !important;
            color: white !important;
            border-radius: 6px;
            border: 1px solid #888;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("fondo.jpeg")

# ---------------- TÍTULO Y LOGO ----------------
st.markdown("<h1 style='text-align: center; color: #A1F55A;'>Chatbot <span style='color:#8BF'>SCIENTIA</span></h1>", unsafe_allow_html=True)
with open("logo2.jpeg", "rb") as image_file:
    logo2_encoded = base64.b64encode(image_file.read()).decode()
st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{logo2_encoded}' width='100'/></div>", unsafe_allow_html=True)
st.markdown("""
<p style='color: white; text-align: center; font-size: 16px;'>
Bienvenido, <strong>Scientia</strong> es un chatbot diseñado para responder tus preguntas sobre el radar de la IA y temas afines. Fórmula tu pregunta abajo.
</p>
""", unsafe_allow_html=True)
# ---------------- CARGA DE DOCUMENTOS ----------------
@st.cache_resource
def load_documents():
    loader = PyPDFLoader("Guión_HMI.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

# ---------------- QA CHAIN ----------------
@st.cache_resource
def setup_qa_chain():
    texts = load_documents()
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key, user_agent="my-app")
    db = TFIDFRetriever.from_documents(texts)

    prompt_template = """
    ## Instructions
    You are an AI personal friendly assistant named Scientia, designed to answer questions about the radar of the IA.
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

# ---------------- INPUT TEXTO ----------------
user_input = st.text_input("✍️ Escribe tu pregunta", key="input")

# ---------------- INPUT AUDIO ----------------
st.markdown("🎙️ <strong>Subir nota de voz</strong>", unsafe_allow_html=True)
audio_file = st.file_uploader("", type=["m4a", "mp3", "wav"], label_visibility="collapsed")

# ---------------- PROCESAMIENTO DE AUDIO ----------------
transcribed_text = None
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🎧 Transcribiendo..."):
        try:
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path, language="Spanish")
            transcribed_text = result["text"]
           
            # LLAMAR AUTOMÁTICAMENTE AL LLM CON EL AUDIO
            with st.spinner("🤖 Pensando..."):
                result = qa_chain.invoke({"query": transcribed_text})
                st.session_state.chat_history.append(("Tú", transcribed_text))
                st.session_state.chat_history.append(("Scientia", result["result"]))
        except Exception as e:
            st.error(f"❌ Error al transcribir: {e}")

# ---------------- BOTÓN ENVIAR TEXTO ----------------
if st.button("Enviar") and user_input.strip():
    with st.spinner("🤖 Pensando..."):
        result = qa_chain.invoke({"query": user_input})
        st.session_state.chat_history.append(("Tú", user_input))
        st.session_state.chat_history.append(("Scientia", result["result"]))

# ---------------- HISTORIAL ----------------
# for speaker, text in st.session_state.chat_history:
for speaker, text in reversed(st.session_state.chat_history):
    icon = "🧑" if speaker == "Tú" else "🤖"
    st.markdown(f"<p style='color: white;'><strong>{icon} {speaker}:</strong> {text}</p>", unsafe_allow_html=True)
