import unittest
from RAG import RAGPipeline



class RAT_TEST(unittest.TestCase):
    def setUp(self):
        self.rag = RAGPipeline(rf'F:\GITHUB\Room-Type-Recommendation-System\static\rooms.json')

    def test_RAG(self):
        question = """
        現代風格，兩人房或三人房
        """
        print(self.rag.getRoomSummaryByRAG(question))

    def test_llm_prediction(self):
        question = """
        現代風格，兩人房或三人房
        """
        rooms_summary = """
        名稱:豪華三人房 價格:6000 面積:33 特色:三張單人床、沙發、茶几 風格:現代 床數:3人房
        名稱:景觀雙人房 價格:5200 面積:28 特色:高樓層景觀、陽台、浴缸 風格:現代 床數:2人房
        名稱:工業風雙人房 價格:5300 面積:27 特色:工業風設計、紅磚牆、裸露吊燈 風格:工業 床數:2人房
        名稱:現代四人房 價格:6800 面積:38 特色:兩張雙人床、沙發、餐桌 風格:現代 床數:4人房
        名稱:現代豪華雙人房 價格:4800 面積:28 特色:大理石浴室、景觀窗、書桌 風格:現代 床數:2人房
        名稱:行政套房 價格:8000 面積:40 特色:商務書桌、沙發區、免費mini bar 風格:現代 床數:2人房
        名稱:湖景套房 價格:7500 面積:36 特色:湖景陽台、浴缸、休憩區 風格:自然 床數:2人房
        名稱:寵物友善房 價格:4500 面積:25 特色:寵物床、戶外空間、寵物用品 風格:自然 床數:2人房
        名稱:和洋混合套房 價格:6200 面積:32 特色:日式榻榻米 + 西式床鋪、茶几 風格:日式/現代 床數:2人房
        名稱:泳池景家庭房 價格:6500 面積:35 特色:泳池景、陽台、家庭桌椅 風格:現代 床數:4人房
        """
        print(self.rag.LLM_Prediction(question, rooms_summary))





if __name__ == '__main__':
    unittest.main()
