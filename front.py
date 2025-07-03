import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.retrievers import TFIDFRetriever
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

# Configurar tu API key
cohere_api_key = '1msKL9N3DxmNqmxMCQLQ4CHz8e1dO130v1urBoUI'

# Cargar el documento
@st.cache_resource
def load_documents():
    loader = PyPDFLoader("Guión_HMI.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

# Crear embeddings y retriever
@st.cache_resource
def setup_qa_chain():
    texts = load_documents()
    
    embeddings = CohereEmbeddings(
        model="multilingual-22-12",
        cohere_api_key=cohere_api_key,
        user_agent="my-app"
    )

    # TF-IDF retriever (sin embeddings por ahora)
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

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Chatbot Científico", layout="centered")
st.title("🤖 Chatbot - Radar de la IA")
st.markdown("Hazle una pregunta sobre el documento PDF cargado.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Tu pregunta", key="input")

if st.button("Enviar") and user_input:
    with st.spinner("Pensando..."):
        result = qa_chain.invoke({"query": user_input})
        st.session_state.chat_history.append(("Tú", user_input))
        st.session_state.chat_history.append(("Scientia", result["result"]))

# Mostrar historial de chat
for speaker, text in st.session_state.chat_history:
    if speaker == "Tú":
        st.markdown(f"**🧑 {speaker}:** {text}")
    else:
        st.markdown(f"**🤖 {speaker}:** {text}")
