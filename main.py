import os
import openai
#import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.prompts import ChatPromptTemplate
#from langchain.memory import ConversationBufferMemory
#from langchain.document_loaders import TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#import shutil
from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Annotated, Optional

from modules.pipe import LLMInitializer
from modules.pipe import EmbeddingInitializer
from modules.pipe import VectorstoreInitializer
from modules.pipe import DocumentProcessor
from modules.pipe import ChainInitializer

app = FastAPI()

load_dotenv()
openai.api_key  = os.environ['OPENAI_API_KEY']

template = """You are an intelligent assistant well-versed in a wide range of topics. 
Your responses should be informative and concise, ideally within three sentences.
If a document is provided, use the information from the document to give an informed answer. 
If no document is provided, use your general knowledge to respond.  
{context}
If a document was provided, the above context contains relevant information. If not, consider the context to be your general knowledge.
Question: {question}
Answer:
"""
llm_initializer = LLMInitializer()
embeddings_initializer = EmbeddingInitializer()
vectorstore_initializer = VectorstoreInitializer()
document_processor = DocumentProcessor()
chain_initializer = ChainInitializer()

conversation_history = []
"""class QueryRequest(BaseModel):
    upload_document: UploadFile = File(...)
    user_query: Annotated[str, Form()]
    selected_llms: List[str]
    temperature: Annotated[float, Form()]
    max_tokens: Annotated[int, Form()]
    top_k: Annotated[int, Form()]
    top_p: float
    selected_embeddings: Annotated[str, Form()]
    selected_vector_databases: Annotated[str, Form()]"""

@app.post("/process_query")
async def process_query(upload_document: UploadFile = File(...),
    user_query: str = Form(...),
    selected_llms: List[str] = Form(...),
    temperature: float = Form(...),
    max_tokens: int = Form(...),
    #top_k: int = Form(...),
    #top_p: float = Form(...),
    selected_embeddings: str = Form(...),
    selected_vector_databases: str = Form(...)):

    def handle_user_selection(upload_document, user_query, selected_llms, temperature, max_tokens, selected_embeddings, selected_vector_databases):
        if "gpt3.5" in selected_llms:
            if "openai" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_openai()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma(chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_gpt(temperature=temperature, max_tokens=max_tokens)
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = rag_chain.invoke(user_query) if retriever else llm.generate(user_query)
            elif "huggingface" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_huggingface()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma(chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_gpt(temperature=temperature, max_tokens=max_tokens)
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = rag_chain.invoke(user_query) #if retriever else llm.generate(user_query)
        elif "Llama" in selected_llms:
            if "openai" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_openai()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma(chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb, k=2)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_llama(temperature=temperature, max_tokens=max_tokens)
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = rag_chain.invoke(user_query) if retriever else llm.generate(user_query)
            elif "huggingface" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_huggingface()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma    (chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb, k=2)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_llama(temperature=temperature, max_tokens=max_tokens)
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = rag_chain.invoke(user_query) if retriever else llm.generate(user_query)
        else:
            result = "Invalid combination selected."

        conversation_history.append((user_query, result))
        return conversation_history
    
    result = handle_user_selection(
        upload_document, user_query, selected_llms,
        temperature, max_tokens, selected_embeddings, selected_vector_databases)
    return result