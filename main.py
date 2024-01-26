import os
import openai
#import gradio as gr
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
#from langchain.memory import ConversationBufferMemory
#from langchain.document_loaders import TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#import shutil

from modules.pipe import initialize_gpt
from modules.pipe import initialize_llama
from modules.pipe import initialize_openai
from modules.pipe import initialize_huggingface
from modules.pipe import initialize_retriever
from modules.pipe import initialize_rag_chain
from modules.pipe import create_and_persist_Chroma
from modules.pipe import process_document

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

conversation_history = []

def process_query(user_query, retriever, llm, rag_chain):
    result = rag_chain.invoke(user_query) if retriever else llm.generate(user_query)
    return result

def handle_user_selection(upload_document, user_query, selected_llms, temperature, max_tokens, top_k, top_p, selected_embeddings, selected_vector_databases):
    query_info = f"Document: {upload_document}\nUser Query: {user_query}\nLLMs: {selected_llms}\nTemperature: {temperature}\nMax_Tokens: {max_tokens}\nTop-k: {top_k}\nTop-p: {top_p}\nEmbeddings: {selected_embeddings}\nVector Databases: {selected_vector_databases}"

    if "gpt3.5" in selected_llms:
        if "openai" in selected_embeddings and "Chroma" in selected_vector_databases:
            embedding = initialize_openai()
            chunks, persist_directory = process_document(upload_document)
            vectordb = create_and_persist_Chroma(chunks, embedding, persist_directory)
            retriever = initialize_retriever(vectordb)
            prompt = ChatPromptTemplate.from_template(template)
            llm = initialize_gpt()
            rag_chain = initialize_rag_chain(retriever, prompt, llm)
            result = process_query(user_query, retriever, llm, rag_chain)
        elif "huggingface" in selected_embeddings and "Chroma" in selected_vector_databases:
            embedding = initialize_huggingface()
            chunks, persist_directory = process_document(upload_document)
            vectordb = create_and_persist_Chroma(chunks, embedding, persist_directory)
            retriever = initialize_retriever(vectordb)
            prompt = ChatPromptTemplate.from_template(template)
            llm = initialize_gpt()
            rag_chain = initialize_rag_chain(retriever, prompt, llm)
            result = process_query(user_query, retriever, llm, rag_chain)
    elif "Llama" in selected_llms:
        if "openai" in selected_embeddings and "Chroma" in selected_vector_databases:
            embedding = initialize_openai()
            chunks, persist_directory = process_document(upload_document)
            vectordb = create_and_persist_Chroma(chunks, embedding, persist_directory)
            retriever = initialize_retriever(vectordb, k=2)
            prompt = ChatPromptTemplate.from_template(template)
            llm = initialize_llama()
            rag_chain = initialize_rag_chain(retriever, prompt, llm)
            result = process_query(user_query, retriever, llm, rag_chain)
        elif "huggingface" in selected_embeddings and "Chroma" in selected_vector_databases:
            embedding = initialize_huggingface()
            chunks, persist_directory = process_document(upload_document)
            vectordb = create_and_persist_Chroma    (chunks, embedding, persist_directory)
            retriever = initialize_retriever(vectordb, k=2)
            prompt = ChatPromptTemplate.from_template(template)
            llm = initialize_llama()
            rag_chain = initialize_rag_chain(retriever, prompt, llm)
            result = process_query(user_query, retriever, llm, rag_chain)
    else:
        result = "Invalid combination selected."

    conversation_history.append((user_query, result))
    return conversation_history

