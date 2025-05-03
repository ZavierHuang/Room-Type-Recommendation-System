import json
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class RAGPipeline:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.docs = [Document(page_content=f"{item['name']}。{item['features']}。") for item in data]

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)

        self.llm = Ollama(model="llama3.2")
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.vectorstore.as_retriever())

    def query(self, question):
        result = self.qa_chain.run(question)
        return result
