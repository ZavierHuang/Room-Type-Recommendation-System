import json
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class RAGPipeline:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.docs = [Document(page_content=f"{item['name']}。{item['features']}。") for item in data]

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)

        self.llm = Ollama(model="llama3.2")
        self.retriever = self.vectorstore.as_retriever()

        self.prompt = PromptTemplate(
            input_variables=["question", "rooms"],
            template=(
                "使用者問題: {question}\n"
                "相關房型: {rooms}\n"
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def query(self, question):
        retrieved_docs = self.retriever.get_relevant_documents(question)
        matched_rooms = []
        for doc in retrieved_docs:
            for room in self.data:
                if room['name'] in doc.page_content:
                    matched_rooms.append(room)
        unique_rooms = {room['name']: room for room in matched_rooms}.values()

        if unique_rooms:
            reply = "可以考慮以下選擇：\n\n"
            for idx, room in enumerate(unique_rooms, 1):
                reply += f"{idx}. {room['name']}（價格：{room['price']}）\n"

            # 組成房型簡短描述給 LLM
            rooms_summary = ', '.join([room['name'] for room in unique_rooms])

            # 請 LLM 生成自然結語
            conclusion = self.chain.run(question=question, rooms=rooms_summary)
            reply += f"\n{conclusion.strip()}"
        else:
            reply = "抱歉，找不到符合你需求的房型，可以換個關鍵字試試喔！"

        return reply
