import streamlit as st
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

# ---------------- FONDO ----------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
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
        /* Estilo botón negro con texto blanco */
        div.stButton > button:first-child {{
            background-color: black !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #555;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("fondo.jpeg")

# ---------------- TÍTULO PRINCIPAL ----------------
st.markdown("<h1 style='text-align: center; color: #A1F55A;'>Chatbot <span style='color:#8BF'>SCIENTIA</span></h1>", unsafe_allow_html=True)

# ---------------- IMAGEN DEL MUÑECO CENTRADA ----------------
with open("logo2.jpeg", "rb") as image_file:
    logo2_encoded = base64.b64encode(image_file.read()).decode()
st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{logo2_encoded}' width='100'/></div>", unsafe_allow_html=True)

# ---------------- TEXTO DE BIENVENIDA ----------------
st.markdown("<h3 style='color: white; text-align: center;'>Bienvenido, Scientia es un chatbot diseñado para responder tus preguntas sobre el radar de la IA y temas afines. Fórmula tu pregunta abajo.</h3>", unsafe_allow_html=True)

# ---------------- CARGA DE DOCUMENTO ----------------
@st.cache_resource
def load_documents():
    loader = PyPDFLoader("Guión_HMI.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

# ---------------- CONFIGURACIÓN QA ----------------
@st.cache_resource
def setup_qa_chain():
    texts = load_documents()

    embeddings = CohereEmbeddings(
        model="multilingual-22-12",
        cohere_api_key=cohere_api_key,
        user_agent="my-app"
    )

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

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "history"]
    )

    llm = Cohere(
        model="command-r-08-2024",
        temperature=0.75,
        cohere_api_key=cohere_api_key,
        max_tokens=300
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db,
        chain_type_kwargs={
            "verbose": False,
            "prompt": PROMPT,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            )
        },
        verbose=False
    )

    return qa

qa_chain = setup_qa_chain()

# ---------------- SESIÓN ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- INPUT Y BOTÓN CENTRADOS ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_input = st.text_input("Tu pregunta", key="input")
    if st.button("Enviar") and user_input:
        with st.spinner("Pensando..."):
            result = qa_chain.invoke({"query": user_input})
            st.session_state.chat_history.append(("Tú", user_input))
            st.session_state.chat_history.append(("Scientia", result["result"]))

# ---------------- HISTORIAL ----------------
for speaker, text in st.session_state.chat_history:
    icon = "🧑" if speaker == "Tú" else "🤖"
    st.markdown(f"<p style='color: white;'><strong>{icon} {speaker}:</strong> {text}</p>", unsafe_allow_html=True)
