import unittest
from src.RAG import RAGPipeline



class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.json_path = r'F:\GITHUB\Room-Type-Recommendation-System\static\rooms.json'
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

    def test_LLM_Prediction_empty(self):
        # 測試 rooms_summary 為空時 LLM_Prediction 邏輯
        result = self.rag.LLM_Prediction('請推薦房型', '')
        self.assertIsInstance(result, str)

    def test_review_recommendation(self):
        result = self.rag.review_recommendation('請推薦房型', '推薦內容')
        self.assertIsInstance(result, str)

    def test_review_recommendation_not_match(self):
        # 模擬 LLM 回傳不符合需求
        result = self.rag.review_recommendation('我要游泳池', '這不是房型推薦')
        self.assertIsInstance(result, str)

    def test_getRoomSummaryByRAG(self):
        summary = self.rag.getRoomSummaryByRAG('我要工業風')
        self.assertIsInstance(summary, str)

    def test_query(self):
        result = self.rag.query('請推薦三人房')
        self.assertIn('conclusion', result)
        self.assertIn('rooms', result)

    def test_query_no_match(self):
        # 測試找不到房型時
        result = self.rag.query('我要一千萬的房型')
        self.assertEqual(result['rooms'], [])

    def test_query_greet(self):
        result = self.rag.query('你好')
        self.assertEqual(result['rooms'], [])
        self.assertIn('我可以幫您推薦合適的房型喔', result['conclusion'])

    def test_query_other(self):
        # 測試 intent 為其他
        self.rag.classify_intent = lambda q: '其他'
        result = self.rag.query('你是誰')
        self.assertEqual(result['rooms'], [])
        self.assertIn('只提供房型相關', result['conclusion'])

    def test_query_empty(self):
        self.rag.query

    def test_auto_recommend_room(self):
        result = self.rag.auto_recommend_room()
        print(result)
        self.assertIn('name', result)
        self.assertIn('price', result)
        self.assertIn('area', result)
        self.assertIn('features', result)
        self.assertIn('style', result)
        self.assertIn('maxOccupancy', result)

    def test_auto_recommend_room_reset(self):
        # 測試 reset_used_names
        self.rag.used_names = {'和式套房'}
        result = self.rag.auto_recommend_room(reset_used_names=True)
        self.assertIn('name', result)
        self.assertEqual(self.rag.used_names, {result['name']})

    def test_auto_recommend_room_prev_name(self):
        # 測試 prev_name
        prev = self.rag.data[0]['name']
        result = self.rag.auto_recommend_room(prev_name=prev)
        self.assertNotEqual(result['name'], prev)

    def test_auto_recommend_room_all_used(self):
        # 測試所有房型都用過 fallback
        self.rag.used_names = set([item['name'] for item in self.rag.data])
        result = self.rag.auto_recommend_room()
        self.assertIn('name', result)

    def test_auto_recommend_room_features_list(self):
        # 測試 features 為 list
        self.rag.data[0]['features'] = ['A', 'B']
        self.rag.used_names = set()
        result = self.rag.auto_recommend_room(reset_used_names=True)
        self.assertIsInstance(result['features'], list)

    def test__parse_max_occupancy(self):
        # Arabic numerals
        self.assertEqual(self.rag._parse_max_occupancy('3人房'), 3)
        self.assertEqual(self.rag._parse_max_occupancy('10人'), 10)
        self.assertEqual(self.rag._parse_max_occupancy('21'), 21)
        # Chinese numerals
        self.assertEqual(self.rag._parse_max_occupancy('三人房'), 3)
        self.assertEqual(self.rag._parse_max_occupancy('十人房'), 10)
        self.assertEqual(self.rag._parse_max_occupancy('十一人房'), 11)
        self.assertEqual(self.rag._parse_max_occupancy('二十人房'), 20)
        self.assertEqual(self.rag._parse_max_occupancy('二十一人房'), 21)
        self.assertEqual(self.rag._parse_max_occupancy('兩人房'), 2)
        self.assertEqual(self.rag._parse_max_occupancy(''), 1)
        self.assertEqual(self.rag._parse_max_occupancy('零人房'), 1)
        self.assertEqual(self.rag._parse_max_occupancy('unknown'), 1)

    def test_query_review_result_branch(self):
        # 模擬 review_recommendation 回傳不符合需求的情境
        room_name = self.rag.data[0]['name']
        review_result = f"目前沒有完全符合的房型，以下是最接近的建議\n房型名稱：{room_name}\n推薦理由：測試"
        self.rag.classify_intent = lambda q: '房型推薦'
        self.rag.getRoomSummaryByRAG = lambda q: self.rag.docs[0].page_content
        self.rag.LLM_Prediction = lambda q, r: 'LLM結論(不會被用到)'
        self.rag.review_recommendation = lambda q, c: review_result
        result = self.rag.query('我要推薦')
        self.assertEqual(result['conclusion'], review_result)
        self.assertIn(room_name, result['rooms'])
        self.assertIn('price', result['rooms'][room_name])
        self.assertIn('area', result['rooms'][room_name])
        self.assertIn('features', result['rooms'][room_name])
        self.assertIn('style', result['rooms'][room_name])
        self.assertIn('maxOccupancy', result['rooms'][room_name])

