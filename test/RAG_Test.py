import unittest
from src.RAG import RAGPipeline



class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.json_path = 'static/rooms.json'
        self.rag = RAGPipeline(self.json_path)

    def test_classify_intent(self):
        self.assertIn(self.rag.classify_intent('請推薦三人房'), ['房型推薦', '打招呼', '其他'])
        self.assertIn(self.rag.classify_intent('你好'), ['房型推薦', '打招呼', '其他'])
        self.assertIn(self.rag.classify_intent('今天天氣如何'), ['房型推薦', '打招呼', '其他'])

    def test_extract_price_range(self):
        self.assertEqual(self.rag.extract_price_range('價格1000~2000元'), (1000, 2000))
        self.assertEqual(self.rag.extract_price_range('2000元以上'), (2000, None))
        self.assertEqual(self.rag.extract_price_range('1500元以下'), (None, 1500))
        self.assertEqual(self.rag.extract_price_range('沒有價格需求'), (None, None))

    def test_extract_area_range(self):
        self.assertEqual(self.rag.extract_area_range('面積30~50m²'), (30, 50))
        self.assertEqual(self.rag.extract_area_range('40平方米以上'), (40, None))
        self.assertEqual(self.rag.extract_area_range('100平方公尺以下'), (None, 100))
        self.assertEqual(self.rag.extract_area_range('沒有面積需求'), (None, None))

    def test_extract_style_keywords(self):
        styles = self.rag.extract_style_keywords('我想要工業風的房型')
        self.assertIsInstance(styles, list)

    def test_sort_by_style_match(self):
        docs = self.rag.docs[:3]
        style_keywords = self.rag.extract_style_keywords('我想要工業風')
        sorted_docs = self.rag.sort_by_style_match(docs, style_keywords)
        self.assertEqual(len(sorted_docs), len(docs))

    def test_filter_by_price_range(self):
        summary = '\n'.join([doc.page_content for doc in self.rag.docs])
        filtered = self.rag.filter_by_price_range(summary, min_price=1000, max_price=3000)
        self.assertIsInstance(filtered, str)

    def test_filter_by_area_range(self):
        summary = '\n'.join([doc.page_content for doc in self.rag.docs])
        filtered = self.rag.filter_by_area_range(summary, min_area=20, max_area=50)
        self.assertIsInstance(filtered, str)

    def test_remove_duplicate_room_names(self):
        conclusion = '房型名稱：A\n推薦理由：好\n房型名稱：A\n推薦理由：重複\n房型名稱：B\n推薦理由：棒'
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertIn('房型名稱：A', result)
        self.assertIn('房型名稱：B', result)
        self.assertNotIn('重複', result)

    def test_LLM_Prediction(self):
        rooms_summary = '\n'.join([doc.page_content for doc in self.rag.docs[:2]])
        result = self.rag.LLM_Prediction('請推薦房型', rooms_summary)
        self.assertIsInstance(result, str)

    def test_review_recommendation(self):
        result = self.rag.review_recommendation('請推薦房型', '推薦內容')
        self.assertIsInstance(result, str)

    def test_getRoomSummaryByRAG(self):
        summary = self.rag.getRoomSummaryByRAG('我要工業風')
        self.assertIsInstance(summary, str)

    def test_query(self):
        result = self.rag.query('請推薦三人房')
        self.assertIn('conclusion', result)
        self.assertIn('rooms', result)

    def test_auto_recommend_room(self):
        result = self.rag.auto_recommend_room()
        print(result)
        self.assertIn('name', result)
        self.assertIn('price', result)
        self.assertIn('area', result)
        self.assertIn('features', result)
        self.assertIn('style', result)
        self.assertIn('maxOccupancy', result)