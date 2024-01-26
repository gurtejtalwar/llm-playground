import datetime

from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma
#from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def process_document(upload_document):
    pdf_loader = PyPDFLoader(upload_document)
    raw_documents = pdf_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_documents)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    persist_directory = f'db_{timestamp}'
    return chunks, persist_directory

def create_and_persist_Chroma(chunks, embedding, persist_directory):
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

def initialize_openai():
    return OpenAIEmbeddings()

def initialize_huggingface():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

def initialize_llama():
    return CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})

def initialize_retriever(vectordb, k=2):
    return vectordb.as_retriever(search_kwargs={'k': k})

def initialize_gpt():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000)

def initialize_rag_chain(retriever, prompt, llm):
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
