import json
import re
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.docstore.document import Document
import json as pyjson
import random

class RAGPipeline:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.docs = [
            Document(page_content=f"名稱:{item['name']} 價格:{item['price']} 面積:{item['area']} 特色:{item['features']} 風格:{item.get('style', '')} 床數:{item.get('maxOccupancy', '')}")
            for item in self.data
        ]

        embeddings = FastEmbedEmbeddings()
        self.vectorstore = Chroma.from_documents(self.docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.llm = Ollama(model="gemma3:27b-it-qat", base_url="http://140.124.184.213:11434")
        self.used_names = set()  # 新增：用於追蹤所有已推薦過的房型名稱


    """
    Classify the user's intent based on their question. (房型推薦 or 打招呼 or 其他)
    """
    def classify_intent(self, question):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "請判斷以下使用者輸入屬於哪一種類型：\n"
             "1. 房型推薦需求（包含價格、風格、幾人房、設備、是否有某項特色等）\n"
             "2. 一般打招呼或聊天（如：你好、哈囉、在嗎）\n"
             "3. 廣泛推薦（如：有什麼推薦、請推薦、推薦一下、可以推薦嗎、請問有推薦的房型嗎等泛用推薦問題，這類問題沒有明確條件，請直接回傳 '泛用推薦'）\n"
             "4. 其他與房型無關的問題（例如問天氣、問你是誰）\n\n"
             "使用者可能會詢問是否有某種房型（例如：是否有工業風？是否有浴缸？），這也屬於房型推薦。\n"
             "請你只回答：'房型推薦'、'打招呼'、'泛用推薦' 或 '其他'"
             ),
            ("user", question)
        ])
        result = (prompt | self.llm).invoke({})
        return result.strip()


    """
    從使用者輸入的內容中提取價格區間，回傳最小價格、最大價格及其嚴格性判斷
    """
    def extract_price_range(self, text):
        min_price = None
        max_price = None
        min_strict = False  # 是否嚴格大於
        max_strict = False  # 是否嚴格小於

        """
        input = "3000~5000元"
        group(1) => 最小價格 = 3000
        group(2) => (元)? (? = 0 or 1 次)
        group(3) => (~|到|至|-|—) 範圍分隔
        group(4) => 最大價格 = 5000
        return (3000, 5000, False, False)
        """
        range_match = re.search(r'(\d{3,5})\s*(元)?\s*(~|到|至|\-|—)\s*(\d{3,5})', text)
        if range_match:
            min_price = int(range_match.group(1))       # 最小價格
            max_price = int(range_match.group(4))       # 最小價格
            return min_price, max_price, min_strict, max_strict

        """
        input = "5000 元起" (>= 5000)
        group(1) => 最小價格 = 3000
        group(2) => (元)
        group(3) => (以上|起|以上的) 
        return (3000, None, False, False)
        
        input = "高於 5000元" (> 5000)
        group(1) => (大於|超過|高於)
        group(2) => 最小價格 = 5000
        group(3) => (元)
        return (5000, None, True, False)
        """
        if match := re.search(r'(\d{3,5})\s*(元)?\s*(以上|起|以上的)', text):
            min_price = int(match.group(1))
            min_strict = False
        elif match := re.search(r'(大於|超過|高於)\s*(\d{3,5})\s*(元)?\s*', text):
            min_price = int(match.group(2))
            min_strict = True

        """
        input = "3000 之內" (<= 3000)
        group(1) => 最高價格 = 3000
        group(2) => (元)
        group(3) => (以上|起|以上的) 
        return (None, 3000, False, False)

        input = "低於 5000元" (< 5000)
        group(1) => (小於|少於|低於)
        group(2) => = 最高價格 5000
        group(3) => (元)
        return (None, 3000, False, True)
        """
        if match := re.search(r'(\d{3,5})\s*(元)?\s*(以下|以內|之內)', text):
            max_price = int(match.group(1))
            max_strict = False
        elif match := re.search(r'(小於|少於|低於)\s*(\d{3,5})\s*(元)?\s*', text):
            max_price = int(match.group(2))
            max_strict = True

        return min_price, max_price, min_strict, max_strict

    """
    從使用者輸入的內容中提取面積區間，回傳最小面積、最大面積及其嚴格性判斷
    """
    def extract_area_range(self, text):
        min_area = None
        max_area = None
        min_strict = False  # 是否嚴格大於
        max_strict = False  # 是否嚴格小於

        # 僅當有面積相關關鍵詞時才進行匹配
        if not re.search(r'(面積|坪|平方|m²|平方米|平方公尺)', text):
            return min_area, max_area, min_strict, max_strict

        """
        input = "面積 50 坪 ~ 100 坪"
        group(1) 最小面積 = 50
        group(2) => (m²|平方公尺|平方米|坪)? (可選)
        group(3) => (~|到|至|-|—) 範圍分隔
        group(4) 最大面積 = 100
        return (50, 100, False, False)
        """
        range_match = re.search(r'(\d{2,4})\s*(m²|平方公尺|平方米|坪)?\s*(~|到|至|\-|—)\s*(\d{2,4})', text)
        if range_match:
            min_area = int(range_match.group(1))
            max_area = int(range_match.group(4))
            return min_area, max_area, False, False

        """
        input = "面積 50 坪以上" (>= 50)
        group(1) 最小面積 = 50
        group(2) => (m²|平方公尺|平方米|坪)? (可選)
        group(3) => (以上|起|以上的)
        return (50, None, False, False)
        
        input = "大於 50 坪" (> 50)
        group(1) => (大於|超過|多於)
        group(2) 最小面積 = 50
        group(3) => (m²|平方公尺|平方米|坪)? (可選)
        return (50, None, True, False)
        """
        if match := re.search(r'(\d{2,4})\s*(m²|平方公尺|平方米|坪)?\s*(以上|起|以上的)', text):
            min_area = int(match.group(1))
            min_strict = False
        elif match := re.search(r'(大於|超過|多於)\s*(\d{2,4})', text):
            min_area = int(match.group(2))
            min_strict = True

        """
        input = "面積 50 坪以下" (<= 50)
        group(1) 最大面積 = 50
        group(2) => (m²|平方公尺|平方米|坪)? (可選)
        group(3) => (以下|以內|之內)
        return (None, 50, False, False)
        
        input = "小於 50 坪" (< 50)
        group(1) => (小於|少於|低於)
        group(2) 最大面積 = 50
        group(3) => (m²|平方公尺|平方米|坪)? (可選)
        return (None, 50, False, True)
        """
        if match := re.search(r'(\d{2,4})\s*(m²|平方公尺|平方米|坪)?\s*(以下|以內|之內)', text):
            max_area = int(match.group(1))
            max_strict = False
        elif match := re.search(r'(小於|少於|低於)\s*(\d{2,4})', text):
            max_area = int(match.group(2))
            max_strict = True

        return min_area, max_area, min_strict, max_strict

    """
    從使用者輸入的內容中提取風格關鍵字，回傳出現在輸入中的所有房型風格
    input = "我要工業風"
    output = ["工業風"]
    """
    def extract_style_keywords(self, text):
        styles = []

        for item in self.data:
            if item["style"] is not None and len(item["style"])!=0 and item["style"] not in styles:
                styles.append(item["style"])

        return [style for style in styles if style is not None and style in text]

    """
    根據風格關鍵字對文件列表進行排序，讓與使用者需求風格相符的房型排在前面
    """
    def sort_by_style_match(self, docs, style_keywords):
        if not style_keywords:
            return docs

        def score(doc):
            content = doc.page_content
            return sum(1 for kw in style_keywords if kw in content)

        return sorted(docs, key=score, reverse=True)

    """
    根據價格區間篩選房型摘要內容。
    參數：
        summary: 房型摘要字串（每行一個房型）
        min_price: 最小價格（可為 None）
        max_price: 最大價格（可為 None）
        min_strict: 是否嚴格大於 min_price
        max_strict: 是否嚴格小於 max_price
    回傳：
        篩選後的房型摘要字串（只保留符合價格條件的房型）
        
    summary = "
        名稱:房A 價格:2000 面積:20
        名稱:房B 價格:3000 面積:30
        名稱:房C 價格:4000 面積:40"
    
    parameter : min_price=3000
    output : "
    名稱:房B 價格:3000 面積:30
    名稱:房C 價格:4000 面積:40"
    """
    def filter_by_price_range(self, summary, min_price=None, max_price=None, min_strict=False, max_strict=False):
        filtered = []
        for doc in summary.split('\n'):
            match = re.search(r'價格[:：]?(\d+)', doc)
            if match:
                price = int(match.group(1))
                if min_price is not None:
                    if min_strict and not (price > min_price):
                        continue
                    if not min_strict and not (price >= min_price):
                        continue
                if max_price is not None:
                    if max_strict and not (price < max_price):
                        continue
                    if not max_strict and not (price <= max_price):
                        continue
                filtered.append(doc)
        return '\n'.join(filtered)

    """
    根據面積區間篩選房型摘要內容。
    參數：
        summary: 房型摘要字串（每行一個房型）
        min_area: 最小面積（可為 None）
        max_area: 最大面積（可為 None）
        min_strict: 是否嚴格大於 min_area
        max_strict: 是否嚴格小於 max_area
    回傳：
        篩選後的房型摘要字串（只保留符合面積條件的房型）
    summary = "
        名稱:房A 價格:2000 面積:20
        名稱:房B 價格:3000 面積:30
        名稱:房C 價格:4000 面積:40"
    parameter : min_area=30
    output : "
        名稱:房B 價格:3000 面積:30
        名稱:房C 價格:4000 面積:40"
    """
    def filter_by_area_range(self, summary, min_area=None, max_area=None, min_strict=False, max_strict=False):
        filtered = []
        for doc in summary.split('\n'):
            match = re.search(r'面積[:：]?(\d{1,4})', doc)
            if match:
                area = int(match.group(1))
                if min_area is not None:
                    if min_strict and not (area > min_area):
                        continue
                    if not min_strict and not (area >= min_area):
                        continue
                if max_area is not None:
                    if max_strict and not (area < max_area):
                        continue
                    if not max_strict and not (area <= max_area):
                        continue
                filtered.append(doc)
        return '\n'.join(filtered)

    """
    移除推薦結論中重複的房型名稱及其推薦理由。
    
    conclusion: 
    房型名稱：豪華雙人房
    推薦理由：空間寬敞
    房型名稱：豪華雙人房
    推薦理由：價格優惠
    房型名稱：經濟單人房
    推薦理由：價格實惠
    結語：歡迎入住
    
    output:
    房型名稱：豪華雙人房
    推薦理由：空間寬敞
    房型名稱：經濟單人房
    推薦理由：價格實惠
    結語：歡迎入住
    """
    def remove_duplicate_room_names(self, conclusion: str):
        seen = set()
        result = []
        lines = conclusion.splitlines()
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if line.startswith("房型名稱："):
                name = line.split("：")[-1].strip()
                if name in seen:
                    skip_next = True
                    continue
                seen.add(name)
            print(line)
            result.append(line)
        return '\n'.join(result)

    def LLM_Prediction(self, question, rooms_summary):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位專業且親切的飯店房型推薦助手，專門根據使用者的需求（例如：預算、風格、入住人數等）提供最合適的房型建議。\n\n"
             "請依據提供的房型資料中，精選出「最符合使用者需求」的房型，最多列出 3 間房型"
             "⚠️ 請注意推薦的房型**不可重複**，若重複則**刪除其中一個房型名稱和推薦理由**。\n"
             "⚠️ 請**『務必只使用資料庫中提供的房型名稱，不可自行編造』。\n"
             "⚠️ 回覆內容請使用**繁體中文**。\n"
             "⚠️ 若使用者的問題與房型推薦無關，請親切回覆：「我是一個飯店推薦助手，目前只提供房型相關的建議喔！」\n"
             "⚠️ 請在推薦理由中明確說明房型的實際價格，並根據使用者預算範圍正確描述：\n"
             "  - 若價格在預算範圍內，請強調『符合預算』或『價格具有優勢』。\n"
             "  - 嚴禁出現『雖然價格稍高』、『超出預算』等與實際價格不符的描述。\n"
             "  - 若房型價格低於預算上限，請強調其價格優勢。\n"
             "⚠️ 若房型未完全符合使用者需求（如面積、價格等），請明確說明『此房型未達到您的需求，但為最接近的選擇』，且不得出現誤導性語句。"
            ),
            ("user",
             "使用者需求：{input}\n"
             "房型資料：{rooms}\n\n"
             "請根據上述資訊，推薦最符合使用者需求的房型，並給出推薦理由與結語。\n"
             "回覆格式如下（每個房型請依照範例填寫）：\n\n"
             "推薦房型：\n"
             "房型名稱\n"
             "推薦理由：...\n\n"
             "房型名稱\n"
             "推薦理由：...\n\n"
             "房型名稱\n"
             "推薦理由：...\n\n"
             "結語：...")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"input": question, "rooms": rooms_summary})
        return result

    """
    將使用者需求與 LLM 輸出的推薦內容一併傳給 LLM，請其判斷推薦內容是否完全符合需求。
    
    輸入：
        user_question: "我要兩人房"
        llm_output: "房型名稱：豪華雙人房\n推薦理由：空間寬敞，價格3000元"
    輸出：
        "推薦內容符合使用者需求，無需變更。"
    
    輸入：
        user_question: "我要四人房"
        llm_output: "房型名稱：雙人房\n推薦理由：價格便宜"
    輸出：
        "目前沒有完全符合的房型"
    """
    def review_recommendation(self, user_question, llm_output):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位專業的飯店房型審查助手。請根據使用者需求與模型原本的推薦內容，判斷是否『完全符合』使用者需求。\n"
             "如果不符合，請回覆：『目前沒有完全符合的房型』\n"
             "請務必使用資料庫中出現過的房型名稱，且回覆內容使用繁體中文。\n"
             "若原本的推薦已經符合需求，則直接回覆：『推薦內容符合使用者需求，無需變更。』"),
            ("user",
             f"使用者需求：{user_question}\n\n模型原本推薦內容如下：\n{llm_output}")
        ])
        chain = prompt | self.llm
        return chain.invoke({})

    """
    根據使用者問題，利用檢索增強生成（RAG）流程取得相關房型摘要。
    1. 先用檢索器取得與問題最相關的房型文件。
    2. 根據使用者輸入中的風格關鍵字，將相關房型排序。
    3. 將排序後的房型摘要內容組合成字串返回。
    
    範例：
      question = "我要工業風雙人房"
      取得相關文件後，若有：
        名稱:現代風雙人房 價格:3200 面積:22 特色:... 風格:現代風 床數:2
        名稱:工業風雙人房 價格:3000 面積:20 特色:... 風格:工業風 床數:2
      則回傳：
        名稱:工業風雙人房 價格:3000 面積:20 特色:... 風格:工業風 床數:2
        名稱:現代風雙人房 價格:3200 面積:22 特色:... 風格:現代風 床數:2
    """
    def getRoomSummaryByRAG(self, question):
        docs = self.retriever.get_relevant_documents(question)
        style_keywords = self.extract_style_keywords(question)
        sorted_docs = self.sort_by_style_match(docs, style_keywords)
        return "\n".join([doc.page_content for doc in sorted_docs])

    """
    處理使用者輸入的主要查詢方法。
    依據使用者輸入自動判斷意圖（如房型推薦、打招呼、其他），並根據意圖給出不同回應：
    - 若為打招呼，回傳歡迎語。
    - 若為房型推薦，會依序：
        1. 取得相關房型摘要
        2. 根據價格、面積等條件過濾
        3. 由 LLM 產生推薦結論
        4. 審查推薦內容是否符合需求
        5. 移除重複房型
        6. 組合回傳房型資訊與結論
    - 其他則回傳預設說明。
    
    範例：
      question = "預算3000元，想要工業風雙人房"
      回傳：
      {
        "rooms": {
          "工業風雙人房": {"price": 3000, "area": 20, ...}
        },
        "conclusion": "推薦房型：\n房型名稱：工業風雙人房...\n結語：..."
      }
    """
    def query(self, question):
        intent = self.classify_intent(question)

        if "打招呼" in intent:
            return {
                "rooms": [],
                "conclusion": "您好，我是一位飯店推薦助手，很高興為您服務！請問您有什麼住宿需求，我可以幫您推薦合適的房型喔～"
            }

        if "泛用推薦" in intent:
            # 隨機取三個房型
            sample_rooms = random.sample(self.data, min(3, len(self.data)))
            conclusion = "推薦房型：\n"
            for item in sample_rooms:
                conclusion += f"房型名稱：{item['name']}\n推薦理由：本房型擁有{item['features']}，價格{item['price']}元，風格為{item['style']}，適合{item['maxOccupancy']}人入住。\n\n"
            conclusion += "結語：以上是我們為您精選的房型，歡迎洽詢！"
            response = {
                "rooms": {item['name']: {
                    "price": item['price'],
                    "area": item['area'],
                    "features": item['features'],
                    "style": item['style'],
                    "maxOccupancy": item['maxOccupancy']
                } for item in sample_rooms},
                "conclusion": conclusion.strip()
            }
            return response

        if "房型推薦" in intent:
            rooms_summary = self.getRoomSummaryByRAG(question)

            min_price, max_price, min_strict, max_strict = self.extract_price_range(question)

            rooms_summary = self.filter_by_price_range(rooms_summary, min_price, max_price, min_strict, max_strict)

            min_area, max_area, min_strict, max_strict = self.extract_area_range(question)
            rooms_summary = self.filter_by_area_range(rooms_summary, min_area, max_area, min_strict, max_strict)

            conclusion = self.LLM_Prediction(question, rooms_summary)
            review_result = self.review_recommendation(question, conclusion)

            if "符合使用者需求" in review_result:
                final_conclusion = conclusion
            else:
                final_conclusion = review_result
            final_conclusion = self.remove_duplicate_room_names(final_conclusion)

            response = {
                "rooms": {},
                "conclusion": final_conclusion
            }

            # 從資料庫中找出推薦的房型並將房型資訊填入回應中
            for item in self.data:
                if item['name'] in final_conclusion:
                    response['rooms'][item['name']] = {
                        "price": item['price'],
                        "area": item['area'],
                        "features": item['features'],
                        "style": item['style'],
                        "maxOccupancy": item['maxOccupancy']
                    }
            return response

        return {
            "rooms": [],
            "conclusion": "你好，我是一個飯店推薦助手，目前只提供房型相關的建議喔！"
        }

    """
    自動推薦一個最適合的房型（不重複），並以 JSON 格式回傳。
    流程：
      1. 依據資料庫內容，請 LLM 推薦一個房型。
      2. 檢查推薦房型名稱是否與資料庫或已推薦過的房型重複。
      3. 若推薦內容格式正確且不重複，則回傳該房型資訊。
      4. 最多重試 5 次，若都失敗則回傳 None。
    
    範例：
      輸出：
      {
        "name": "工業風雙人房",
        "price": 3000,
        "area": 20,
        "features": "工業風設計，附浴缸",
        "style": "工業風",
        "maxOccupancy": 2
      }
    """
    def auto_recommend_room(self):
        required_fields = ['name', 'price', 'area', 'features', 'style', 'maxOccupancy']
        max_retry = 5

        self.used_names = set()

        existing_names = {room['name'] for room in self.data if 'name' in room}
        self.used_names.update(existing_names)

        for _ in range(max_retry):
            system_msg = (
                "你是一位專業的飯店房型推薦助手，請根據資料庫內容，推薦一個最適合的房型，並只回傳一個 JSON 格式，欄位包含 name, price, area, features, style, maxOccupancy。"
                "area、price、maxOccupancy只回傳數字，剩下內容請轉為「繁體中文」，不要有多餘說明。"
                "請勿推薦與前次相同或相似的房型名稱，也不要與資料庫中已有的房型名稱重複。"
                "房型名稱需能明確反映其特色（如特色設施、風格等），讓名稱與特色相對應。"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("user", "請推薦一個房型，並只回傳 JSON 格式資料。")
            ])
            chain = prompt | self.llm
            result = chain.invoke({})

            # 抓取 result 中 第一個從 { 到 } 的內容
            # re.DOTALL : 完整擷取多行 JSON 區塊
            match = re.search(r'\{.*\}', result, re.DOTALL)

            if not match:
                continue

            json_str = match.group(0)
            try:
                # 使用 pyjson 解析成 Python 字典
                data = pyjson.loads(json_str)
            except pyjson.JSONDecodeError:
                continue

            """
            - 確認 JSON 有所有欄位 且內容非空
            - 名稱不可重複
            - 通過驗證則回傳這個房型資料
            """
            if all(field in data and data[field] for field in required_fields):
                name = data['name']
                if name in self.used_names:
                    continue
                # 呼叫 _parse_max_occupancy() 確保其為數字
                data['maxOccupancy'] = self._parse_max_occupancy(data['maxOccupancy'])
                self.used_names.add(name)
                return {k: data[k] for k in required_fields}

        return None

    """
    將中文數字或數字字串轉換為整數型態的最大入住人數。
    支援阿拉伯數字（如 2、4）、中文數字（如「兩」、「三」、「十」等）及複合中文數字（如「十二」、「二十」）。
    若無法解析則預設回傳 1。
    
    範例：
      value："2"，return：2
      value："兩人"，return：2
      value："十"，return：10
      value："十二人"，return：12
      value："二十"，return：20
      value："三"，return：3
      value："abc"，return：1
    """
    def _parse_max_occupancy(self, value):
        zh2num = {'零':0,'一':1,'二':2,'兩':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
        value = str(value)

        """
        優先處理阿拉伯數字
        找到數字就直接回傳
        """
        match = re.search(r'\d+', value)
        if match:
            return int(match.group(0))
        num = 0
        if value.startswith('十'):
            # ex: value = "十二"
            num = 10
            if len(value) > 1 and value[1] in zh2num:
                num += zh2num[value[1]]
            return num
        elif '十' in value:
            # ex: value = "二十一"
            # parts = ["二", "一"]
            parts = value.split('十')
            if parts[0] in zh2num:
                num = zh2num[parts[0]] * 10
            if len(parts) > 1 and parts[1] and parts[1][0] in zh2num:
                num += zh2num[parts[1][0]]
            return num
        else:
            # ex: value = "兩人"
            for k in zh2num:
                if k in value:
                    return zh2num[k]
        return  1
