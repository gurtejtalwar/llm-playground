import datetime
import tempfile
import os
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

class DocumentProcessor:
    def process_document(self, upload_document):
        self.upload_document = upload_document
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, upload_document.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(upload_document.file.read())
        print(temp_file_path)
        pdf_loader = PyPDFLoader(temp_file_path)
        raw_documents = pdf_loader.load()
        print("Documents loaded")
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(raw_documents)
        print("Documents split")
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.persist_directory = f'VectorDB/db_{timestamp}'
        os.remove(temp_file_path)
        os.rmdir(temp_dir)
        return chunks, self.persist_directory

    def create_and_persist_Chroma(self, chunks, embedding, persist_directory):
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)
        print ("Chroma Created")
        vectordb.persist()
        print ("Chroma Created and Persisted")
        return Chroma(persist_directory=self.persist_directory, embedding_function=embedding)

class EmbeddingInitializer:
    @staticmethod
    def initialize_openai(): 
        return OpenAIEmbeddings()

    @staticmethod
    def initialize_huggingface():
        return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

class VectorstoreInitializer:
    @staticmethod
    def initialize_retriever(vectordb, k=2):
        return vectordb.as_retriever(search_kwargs={'k': k})

class LLMInitializer:
    @staticmethod
    def initialize_gpt():
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000)

    @staticmethod
    def initialize_llama():
        return CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})

class ChainInitializer:
    @staticmethod
    def initialize_rag_chain(retriever, prompt, llm):
        return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
