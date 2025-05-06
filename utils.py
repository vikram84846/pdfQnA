from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import  ChatPromptTemplate


load_dotenv()

def embeddingModel():
    model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return model


def store_embeddings(loader,filepath):
    pages = loader.load()
    for page in pages:
        page.metadata["source"]= os.path.basename(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    vector_db = Chroma.from_documents(documents=docs,embedding=embeddingModel(),persist_directory="chroma_db")


def get_retriver(filepath):
    vector_db = Chroma(
        persist_directory="chroma_db",
        embedding_function= embeddingModel()
    )
    retriver = vector_db.as_retriever(
        search_type ="similarity",
        search_kwargs ={"k":3,
                        "filter":{"source":os.path.basename(filepath)}
                        }

    )
    return retriver


def get_prompt():
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
    return prompt

def chatmodel():
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temp=0.5,max_tokens= 500)
    return chat_model