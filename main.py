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
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional

from modules.pipe import LLMInitializer
from modules.pipe import EmbeddingInitializer
from modules.pipe import VectorstoreInitializer
from modules.pipe import DocumentProcessor
from modules.pipe import ChainInitializer

app = FastAPI()

load_dotenv()
openai.api_key = 'YOUR_OPENAPI_KEY'
os.environ['OPENAI_API_KEY'] = openai.api_key
print(openai.api_key)

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
class QueryRequest(BaseModel):
    upload_document: UploadFile = File(...)
    user_query: str
    selected_llms: List[str]
    temperature: float
    max_tokens: int
    top_k: int
    top_p: float
    selected_embeddings: str
    selected_vector_databases: str

@app.post("/process_query")
async def process_query(request: QueryRequest):

#def process_query(user_query, retriever, llm, rag_chain):
#    result = rag_chain.invoke(user_query) if retriever else llm.generate(user_query)
#    return result
    def handle_user_selection(upload_document, user_query, selected_llms, temperature, max_tokens, top_k, top_p, selected_embeddings, selected_vector_databases):
        query_info = f"Document: {upload_document}\nUser Query: {user_query}\nLLMs: {selected_llms}\nTemperature: {temperature}\nMax_Tokens: {max_tokens}\nTop-k: {top_k}\nTop-p: {top_p}\nEmbeddings: {selected_embeddings}\nVector Databases: {selected_vector_databases}"
        if "gpt3.5" in selected_llms:
            if "openai" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_openai()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma(chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_gpt()
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = process_query(user_query, retriever, llm, rag_chain)
            elif "huggingface" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_huggingface()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma(chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_gpt()
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = process_query(user_query, retriever, llm, rag_chain)
        elif "Llama" in selected_llms:
            if "openai" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_openai()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma(chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb, k=2)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_llama()
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = process_query(user_query, retriever, llm, rag_chain)
            elif "huggingface" in selected_embeddings and "Chroma" in selected_vector_databases:
                embedding = embeddings_initializer.initialize_huggingface()
                chunks, persist_directory = document_processor.process_document(upload_document)
                vectordb = document_processor.create_and_persist_Chroma    (chunks, embedding, persist_directory)
                retriever = vectorstore_initializer.initialize_retriever(vectordb, k=2)
                prompt = ChatPromptTemplate.from_template(template)
                llm = llm_initializer.initialize_llama()
                rag_chain = chain_initializer.initialize_rag_chain(retriever, prompt, llm)
                result = process_query(user_query, retriever, llm, rag_chain)
        else:
            result = "Invalid combination selected."

        conversation_history.append((user_query, result))
        return conversation_history



