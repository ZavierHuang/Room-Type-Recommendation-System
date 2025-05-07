import json
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class RAGPipeline:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # 把房型資料串接起來作為 embedding corpus
        self.docs = [
            Document(page_content=f"名稱:{item['name']} 價格:{item['price']} 面積:{item['area']} 特色:{item['features']} 風格:{item.get('style', '')} 床數:{item.get('maxOccupancy', '')}")
            for item in self.data
        ]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

    def LLM_Prediction(self, question, rooms_summary):
        llm = Ollama(model="llama3.2:latest", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "你是一個專業且親切的飯店推薦助手，請根據使用者的需求（例如價格、風格、幾人房）\n"
            "請根據以上提供的房型資料中挑選出「最符合」客戶需求的房型，且最多只列出3個。\n"
            "如果使用者的問題與房型推薦無關，請禮貌回覆「我是一個飯店推薦助手，目前只提供房型相關的建議喔！」。\n"
            "不要編造資料庫中沒有的房型名稱，請務必使用「繁體中文」回覆。\n"),
            ("user",
            "使用者需求：{input}\n"
            "房型資料：{rooms}\n"
            "請給出推薦房型及結語，房型名稱務必與資料庫中相同。\n"
            "回覆格式如下：\n"
            "推薦房型：\n"
            "房型名稱\n"
            "推薦理由：...\n"
            "結語：..."
            )
        ])


        chain = prompt | llm
        result = chain.invoke({"input": question, "rooms": rooms_summary})
        return result

    def getRoomSummaryByRAG(self,question):
        # 只用 retriever 撈 top10，不做 Python 過濾
        docs = self.retriever.get_relevant_documents(question)
        # 組成房型摘要文字
        return "\n".join([doc.page_content for doc in docs])


    def query(self, question):
        rooms_summary = self.getRoomSummaryByRAG(question)
        if rooms_summary.strip():
            conclusion = self.LLM_Prediction(question, rooms_summary)
            response = {
                "rooms": {},  # LLM 自己負責選擇和輸出
                "conclusion": conclusion
            }

            for item in self.data:
                if item['name'] in conclusion:
                    response['rooms'][item['name']] = {
                        "price" : item['price'],
                        "area": item['area'],
                        "features": item['features'],
                        "style": item['style'],
                        "maxOccupancy": item['maxOccupancy']
                    }

            print("conclusion:", conclusion)

        else:
            response = {
                "rooms": [],
                "conclusion": "抱歉，找不到符合您需求的房型，可以換個關鍵字再試試喔！"
            }



        print("response:", response)
        return response
