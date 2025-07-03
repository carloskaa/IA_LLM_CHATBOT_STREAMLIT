from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.retrievers import TFIDFRetriever
from langchain.memory import ConversationBufferMemory


# Cargar el documento de la carpeta docs (contenido). Se carga el Guión (Guión_HMI)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFDirectoryLoader("/cotenido/")
loader = PyPDFLoader("Guión_HMI.pdf")  # reemplaza con el nombre real de tu PDF

docs = loader.load()

# Aplicar Split sobre el texto para separarlo en badges

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
len(texts)

# Uso de la API KEY de Cohere para el chatbot (no cambiar)

cohere_api_key='1msKL9N3DxmNqmxMCQLQ4CHz8e1dO130v1urBoUI'

# Crear el embedding con el modelo libre "multilingual-22-12" de Cohere para su uso en español. No está en uso pero hizo parte de las pruebas.

embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=cohere_api_key,user_agent="my-app"
    )


# Creación del contexto para el chatbot

#db = Qdrant.from_documents(texts, embeddings, location=":memory:", collection_name="test", distance_func="Dot")

db = TFIDFRetriever.from_documents(texts)


#Configuración del Prompt (En inglés para mayor precisión). Incluye instrucciones, contexto (Guión), pregunta (usuario) e historial de chat.

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
    template=prompt_template, input_variables=["context", "question", "history"]
)


# Creación del chatbot. El modelo usado es "command-r-08-2024" por tener las mejores cualidades en español. Incluye el Prompt creado.

qa = RetrievalQA.from_chain_type(llm=Cohere(model="command-r-08-2024", temperature=0.75,cohere_api_key=cohere_api_key,max_tokens=300),
                                 chain_type="stuff",
                                 retriever=db,
                                 verbose=False,
                                 chain_type_kwargs = {"verbose": False,"prompt": PROMPT,
                                                      "memory": ConversationBufferMemory(
                                                                memory_key="history",
                                                                input_key="question"),})


#Prueba de respuesta

answer = qa.invoke({"query": "¿Qué es el radar de la IA?"})
print(answer)