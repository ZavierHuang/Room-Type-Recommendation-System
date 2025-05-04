import json
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class RAGPipeline:
    def __init__(self, json_path):
        # 讀入 rooms.json
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.docs = [Document(page_content=f"{item['name']}。{item['features']}。") for item in self.data]

        # 初始化向量資料庫
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def LLM_Prediction(self, question, rooms_summary):
        llm = Ollama(
            model="llama3.2:latest",
            temperature=0.5
        )

        # 設計 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一個親切的飯店客服，會根據提供的房型推薦給客人，結語請自然、熱情，不要自己編造新房型名稱。"),
            ("user", "使用者問題：{input}\n相關房型：{rooms}\n請給出一段推薦的結語，並且使用繁體中文回覆。")
        ])

        # 建立 chain
        chain = prompt | llm
        result = chain.invoke({"input": question, "rooms": rooms_summary})

        

        return result

    def query(self, question):
        # 用 retriever 找房型
        retrieved_docs = self.retriever.get_relevant_documents(question)
        matched_rooms = []
        for doc in retrieved_docs:
            for room in self.data:
                if room['name'] in doc.page_content:
                    matched_rooms.append(room)
        unique_rooms = {room['name']: room for room in matched_rooms}.values()

        if unique_rooms:
            rooms_list = list(unique_rooms)
            rooms_summary = "、".join([room['name'] for room in rooms_list])

            # 由 LLM 生成結語
            conclusion = self.LLM_Prediction(question, rooms_summary)

            # 組合回傳結果（房型列表 + 結語）
            response = {
                "rooms": rooms_list,
                "conclusion": conclusion
            }
        else:
            response = {
                "rooms": [],
                "conclusion": "抱歉，找不到符合你需求的房型，可以換個關鍵字試試喔！"
            }

        print("response:", response)

        return response
