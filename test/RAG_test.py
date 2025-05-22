import unittest
from unittest.mock import patch, MagicMock
from src.RAG import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.llm = MagicMock()

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_classify_intent_room_recommend(self, mock_prompt):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '房型推薦'
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.classify_intent('請推薦三人房')
        self.assertEqual(result, '房型推薦')

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_classify_intent_greeting(self, mock_prompt):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '打招呼'
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.classify_intent('你好')
        self.assertEqual(result, '打招呼')

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_classify_intent_other(self, mock_prompt):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '其他'
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.classify_intent('今天天氣如何？')
        self.assertEqual(result, '其他')

    def test_range_price_pattern(self):
        # 測試各種區間格式
        self.assertEqual(self.rag.extract_price_range("價格2000~3000元"), (2000, 3000))
        self.assertEqual(self.rag.extract_price_range("1500到2500"), (1500, 2500))
        self.assertEqual(self.rag.extract_price_range("1000-2000"), (1000, 2000))
        self.assertEqual(self.rag.extract_price_range("1200—1800"), (1200, 1800))
        self.assertEqual(self.rag.extract_price_range("3000 至 5000元"), (3000, 5000))

    def test_min_price_pattern(self):
        self.assertEqual(self.rag.extract_price_range("2000元以上"), (2000, None))
        self.assertEqual(self.rag.extract_price_range("2000元以上的房型"), (2000, None))
        self.assertEqual(self.rag.extract_price_range("2000元起的房型"), (2000, None))

    def test_max_price_pattern(self):
        self.assertEqual(self.rag.extract_price_range("2000元以內"), (None, 2000))
        self.assertEqual(self.rag.extract_price_range("2000元以下"), (None, 2000))
        self.assertEqual(self.rag.extract_price_range("2000元之內"), (None, 2000))

    def test_range_area_pattern(self):
        # 測試各種區間格式
        self.assertEqual(self.rag.extract_area_range("30~50m²"), (30, 50))
        self.assertEqual(self.rag.extract_area_range("30至50平方公尺"), (30, 50))
        self.assertEqual(self.rag.extract_area_range("20~60"), (20, 60))
        self.assertEqual(self.rag.extract_area_range("1200—1800"), (1200, 1800))
        self.assertEqual(self.rag.extract_area_range("40平方米~60平方米"), (40, 60))

    def test_min_area_pattern(self):
        self.assertEqual(self.rag.extract_area_range("40平方公尺以上的房型"), (40, None))
        self.assertEqual(self.rag.extract_area_range("面積40起的房型"), (40, None))
        self.assertEqual(self.rag.extract_area_range("40m²以上"), (40, None))

    def test_max_area_pattern(self):
        self.assertEqual(self.rag.extract_area_range("40平方公尺以下的房型"), (None, 40))
        self.assertEqual(self.rag.extract_area_range("面積40以內的房型"), (None, 40))
        self.assertEqual(self.rag.extract_area_range("40m²以下"), (None, 40))

    def test_extract_style_keywords_basic(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"style": "工業風"},
            {"style": "北歐風"},
            {"style": "現代風"},
            {"style": "工業風"},  # 重複
        ]
        # text 同時包含多個 style
        text = "我想要工業風或北歐風的房型"
        result = self.rag.extract_style_keywords(text)
        self.assertEqual(set(result), {"工業風", "北歐風"})

    def test_extract_style_keywords_no_match(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"style": "工業風"},
            {"style": "北歐風"},
        ]
        text = "我想要日式風格的房型"
        result = self.rag.extract_style_keywords(text)
        self.assertEqual(result, [])

    def test_extract_style_keywords_empty_and_none(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"style": "工業風"},
            {"style": ""},
            {"style": None},
            {"style": "現代風"},
        ]
        text = "現代風的房型推薦"
        result = self.rag.extract_style_keywords(text)
        print("result:",result)
        self.assertEqual(result, ["現代風"])

    def test_extract_style_keywords_all_styles(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"style": "工業風"},
            {"style": "北歐風"},
            {"style": "現代風"},
        ]
        text = "我想要工業風、北歐風、現代風的房型"
        result = self.rag.extract_style_keywords(text)
        self.assertEqual(set(result), {"工業風", "北歐風", "現代風"})

    def test_sort_by_style_match_no_keywords(self):
        # 若 style_keywords 為空，應直接回傳原 docs
        docs = [MagicMock(page_content="A"), MagicMock(page_content="B")]
        result = self.rag.sort_by_style_match(docs, [])
        self.assertEqual(result, docs)

    def test_sort_by_style_match_with_keywords(self):
        # 測試排序是否正確
        doc1 = MagicMock(page_content="工業風 北歐風")  # 2 matches
        doc2 = MagicMock(page_content="工業風")        # 1 match
        doc3 = MagicMock(page_content="現代風")        # 0 match
        docs = [doc3, doc2, doc1]
        style_keywords = ["工業風", "北歐風"]
        result = self.rag.sort_by_style_match(docs, style_keywords)
        self.assertEqual(result, [doc1, doc2, doc3])

    def test_sort_by_style_match_same_score(self):
        # 若分數相同，順序應與原本一致（穩定排序）
        doc1 = MagicMock(page_content="工業風")
        doc2 = MagicMock(page_content="北歐風")
        docs = [doc1, doc2]
        style_keywords = ["工業風", "北歐風"]
        result = self.rag.sort_by_style_match(docs, style_keywords)
        self.assertEqual(result, docs)

    def test_sort_by_style_match_partial_match(self):
        # 只部分匹配
        doc1 = MagicMock(page_content="工業風")
        doc2 = MagicMock(page_content="無風格")
        docs = [doc2, doc1]
        style_keywords = ["工業風"]
        result = self.rag.sort_by_style_match(docs, style_keywords)
        self.assertEqual(result, [doc1, doc2])

    def test_filter_by_price_range_all_none(self):
        # min_price, max_price 都為 None，應回傳所有有價格的行
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_price_range(summary)
        self.assertEqual(result, summary)

    def test_filter_by_price_range_min(self):
        # 只設 min_price
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_price_range(summary, min_price=3000)
        self.assertEqual(result, "名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40")

    def test_filter_by_price_range_max(self):
        # 只設 max_price
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_price_range(summary, max_price=2500)
        self.assertEqual(result, "名稱:房A 價格:2000 面積:20")

    def test_filter_by_price_range_min_and_max(self):
        # 同時設 min_price, max_price
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_price_range(summary, min_price=2000, max_price=3000)
        self.assertEqual(result, "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30")

    def test_filter_by_price_range_no_match(self):
        # 沒有任何房型符合
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30"
        result = self.rag.filter_by_price_range(summary, min_price=4000)
        self.assertEqual(result, "")

    def test_filter_by_price_range_no_price(self):
        # 行內沒有價格資訊
        summary = "名稱:房A 面積:20\n名稱:房B 面積:30"
        result = self.rag.filter_by_price_range(summary, min_price=1000)
        self.assertEqual(result, "")

    def test_filter_by_area_range_all_none(self):
        # min_area, max_area 都為 None，應回傳所有有面積的行
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_area_range(summary)
        self.assertEqual(result, summary)

    def test_filter_by_area_range_min(self):
        # 只設 min_area
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_area_range(summary, min_area=30)
        self.assertEqual(result, "名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40")

    def test_filter_by_area_range_max(self):
        # 只設 max_area
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_area_range(summary, max_area=25)
        self.assertEqual(result, "名稱:房A 價格:2000 面積:20")

    def test_filter_by_area_range_min_and_max(self):
        # 同時設 min_area, max_area
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30\n名稱:房C 價格:4000 面積:40"
        result = self.rag.filter_by_area_range(summary, min_area=20, max_area=30)
        self.assertEqual(result, "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30")

    def test_filter_by_area_range_no_match(self):
        # 沒有任何房型符合
        summary = "名稱:房A 價格:2000 面積:20\n名稱:房B 價格:3000 面積:30"
        result = self.rag.filter_by_area_range(summary, min_area=50)
        self.assertEqual(result, "")

    def test_filter_by_area_range_no_area(self):
        # 行內沒有面積資訊
        summary = "名稱:房A 價格:2000\n名稱:房B 價格:3000"
        result = self.rag.filter_by_area_range(summary, min_area=10)
        self.assertEqual(result, "")

    def test_remove_duplicate_room_names_none(self):
        # 沒有重複房型名稱
        conclusion = "房型名稱：A\n推薦理由：好\n房型名稱：B\n推薦理由：棒\n結語：歡迎入住"
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertEqual(result, conclusion)

    def test_remove_duplicate_room_names_with_duplicate(self):
        # 有重複房型名稱，應移除重複的房型名稱及其推薦理由
        conclusion = "房型名稱：A\n推薦理由：好\n房型名稱：B\n推薦理由：棒\n房型名稱：A\n推薦理由：再推一次\n結語：歡迎入住"
        expected = "房型名稱：A\n推薦理由：好\n房型名稱：B\n推薦理由：棒\n結語：歡迎入住"
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertEqual(result, expected)

    def test_remove_duplicate_room_names_duplicate_at_end(self):
        # 重複房型名稱在結語前
        conclusion = "房型名稱：A\n推薦理由：好\n房型名稱：A\n推薦理由：再推一次\n結語：歡迎入住"
        expected = "房型名稱：A\n推薦理由：好\n結語：歡迎入住"
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertEqual(result, expected)

    def test_remove_duplicate_room_names_consecutive_duplicates(self):
        # 連續重複房型名稱
        conclusion = "房型名稱：A\n推薦理由：好\n房型名稱：A\n推薦理由：再推一次\n房型名稱：A\n推薦理由：三推\n結語：歡迎入住"
        expected = "房型名稱：A\n推薦理由：好\n結語：歡迎入住"
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertEqual(result, expected)

    def test_remove_duplicate_room_names_only_duplicates(self):
        # 只有重複房型名稱
        conclusion = "房型名稱：A\n推薦理由：好\n房型名稱：A\n推薦理由：再推一次"
        expected = "房型名稱：A\n推薦理由：好"
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertEqual(result, expected)

    def test_remove_duplicate_room_names_no_room(self):
        # 沒有任何房型名稱
        conclusion = "推薦理由：好\n結語：歡迎入住"
        result = self.rag.remove_duplicate_room_names(conclusion)
        self.assertEqual(result, conclusion)

    def test_getRoomSummaryByRAG_basic(self):
        # 準備 mock retriever, extract_style_keywords, sort_by_style_match
        self.rag = RAGPipeline.__new__(RAGPipeline)
        mock_doc1 = MagicMock(page_content="名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2")
        mock_doc2 = MagicMock(page_content="名稱:房B 價格:3000 面積:30 特色:小 風格:北歐風 床數:3")
        self.rag.retriever = MagicMock()
        self.rag.retriever.get_relevant_documents.return_value = [mock_doc1, mock_doc2]
        self.rag.extract_style_keywords = MagicMock(return_value=["工業風"])
        self.rag.sort_by_style_match = MagicMock(return_value=[mock_doc1, mock_doc2])
        result = self.rag.getRoomSummaryByRAG("我要工業風")
        self.assertEqual(result, "名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2\n名稱:房B 價格:3000 面積:30 特色:小 風格:北歐風 床數:3")
        self.rag.retriever.get_relevant_documents.assert_called_once_with("我要工業風")
        self.rag.extract_style_keywords.assert_called_once_with("我要工業風")
        self.rag.sort_by_style_match.assert_called_once_with([mock_doc1, mock_doc2], ["工業風"])

    def test_getRoomSummaryByRAG_empty_docs(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.retriever = MagicMock()
        self.rag.retriever.get_relevant_documents.return_value = []
        self.rag.extract_style_keywords = MagicMock(return_value=[])
        self.rag.sort_by_style_match = MagicMock(return_value=[])
        result = self.rag.getRoomSummaryByRAG("隨便問")
        self.assertEqual(result, "")

    def test_getRoomSummaryByRAG_style_keywords_none(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        mock_doc = MagicMock(page_content="名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2")
        self.rag.retriever = MagicMock()
        self.rag.retriever.get_relevant_documents.return_value = [mock_doc]
        self.rag.extract_style_keywords = MagicMock(return_value=[])
        self.rag.sort_by_style_match = MagicMock(return_value=[mock_doc])
        result = self.rag.getRoomSummaryByRAG("沒有風格")
        self.assertEqual(result, "名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2")

    def test_getRoomSummaryByRAG_sorting_effect(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        mock_doc1 = MagicMock(page_content="名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2")
        mock_doc2 = MagicMock(page_content="名稱:房B 價格:3000 面積:30 特色:小 風格:北歐風 床數:3")
        self.rag.retriever = MagicMock()
        self.rag.retriever.get_relevant_documents.return_value = [mock_doc1, mock_doc2]
        self.rag.extract_style_keywords = MagicMock(return_value=["北歐風"])
        # 測試排序效果
        self.rag.sort_by_style_match = MagicMock(return_value=[mock_doc2, mock_doc1])
        result = self.rag.getRoomSummaryByRAG("我要北歐風")
        self.assertEqual(result, "名稱:房B 價格:3000 面積:30 特色:小 風格:北歐風 床數:3\n名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2")

    def test_parse_max_occupancy_arabic(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.assertEqual(self.rag._parse_max_occupancy('3'), 3)
        self.assertEqual(self.rag._parse_max_occupancy('10'), 10)
        self.assertEqual(self.rag._parse_max_occupancy('25人'), 25)
        self.assertEqual(self.rag._parse_max_occupancy('100'), 100)

    def test_parse_max_occupancy_chinese_simple(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.assertEqual(self.rag._parse_max_occupancy('一'), 1)
        self.assertEqual(self.rag._parse_max_occupancy('兩'), 2)
        self.assertEqual(self.rag._parse_max_occupancy('三'), 3)
        self.assertEqual(self.rag._parse_max_occupancy('十'), 10)

    def test_parse_max_occupancy_chinese_compound(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.assertEqual(self.rag._parse_max_occupancy('十一'), 11)
        self.assertEqual(self.rag._parse_max_occupancy('二十'), 20)
        self.assertEqual(self.rag._parse_max_occupancy('二十三'), 23)
        self.assertEqual(self.rag._parse_max_occupancy('十七'), 17)

    def test_parse_max_occupancy_invalid(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.assertEqual(self.rag._parse_max_occupancy(''), 1)
        self.assertEqual(self.rag._parse_max_occupancy(None), 1)
        self.assertEqual(self.rag._parse_max_occupancy('未知'), 1)
        self.assertEqual(self.rag._parse_max_occupancy('abc'), 1)

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_success(self, mock_prompt):
        # 模擬 LLM 回傳正確 JSON 格式且所有欄位皆有
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"name": "A", "price": 1000, "area": 10, "features": "大", "style": "工業風", "maxOccupancy": 2}
        ]
        self.rag.used_names = set()
        self.rag.llm = MagicMock()  # 修正：補上 llm mock
        mock_chain = MagicMock()
        # LLM 回傳內容
        mock_chain.invoke.return_value = '{"name": "B", "price": 2000, "area": 20, "features": "小", "style": "北歐風", "maxOccupancy": "3"}'
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertEqual(result, {
            "name": "B", "price": 2000, "area": 20, "features": "小", "style": "北歐風", "maxOccupancy": 3
        })

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_duplicate_name(self, mock_prompt):
        # LLM 回傳重複名稱，應重試
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"name": "A", "price": 1000, "area": 10, "features": "大", "style": "工業風", "maxOccupancy": 2}
        ]
        self.rag.used_names = set()
        self.rag.llm = MagicMock()  # 修正：補上 llm mock
        mock_chain = MagicMock()
        # 先回傳重複名稱，再回傳新名稱
        mock_chain.invoke.side_effect = [
            '{"name": "A", "price": 2000, "area": 20, "features": "小", "style": "北歐風", "maxOccupancy": "3"}',
            '{"name": "B", "price": 3000, "area": 30, "features": "中", "style": "現代風", "maxOccupancy": "4"}'
        ]
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertEqual(result, {
            "name": "B", "price": 3000, "area": 30, "features": "中", "style": "現代風", "maxOccupancy": 4
        })

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_invalid_json(self, mock_prompt):
        # LLM 回傳無法解析的 JSON，應重試
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = []
        self.rag.used_names = set()
        self.rag.llm = MagicMock()  # 修正：補上 llm mock
        mock_chain = MagicMock()
        # 先回傳無法解析，再回傳正確
        mock_chain.invoke.side_effect = [
            '這不是json',
            '{"name": "C", "price": 4000, "area": 40, "features": "大", "style": "日式", "maxOccupancy": "5"}'
        ]
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertEqual(result, {
            "name": "C", "price": 4000, "area": 40, "features": "大", "style": "日式", "maxOccupancy": 5
        })

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_missing_field(self, mock_prompt):
        # LLM 回傳缺少欄位，應重試
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = []
        self.rag.used_names = set()
        self.rag.llm = MagicMock()  # 修正：補上 llm mock
        mock_chain = MagicMock()
        # 先回傳缺欄位，再回傳正確
        mock_chain.invoke.side_effect = [
            '{"name": "D", "price": 5000, "area": 50, "features": "大", "style": "美式"}',
            '{"name": "E", "price": 6000, "area": 60, "features": "大", "style": "現代", "maxOccupancy": "6"}'
        ]
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertEqual(result, {
            "name": "E", "price": 6000, "area": 60, "features": "大", "style": "現代", "maxOccupancy": 6
        })

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_max_retry(self, mock_prompt):
        # LLM 連續回傳無效，最終回傳 None
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = []
        self.rag.used_names = set()
        self.rag.llm = MagicMock()  # 修正：補上 llm mock
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '這不是json'
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertIsNone(result)

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_jsondecodeerror(self, mock_prompt):
        # LLM 回傳格式錯誤，觸發 JSONDecodeError，應重試
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = []
        self.rag.used_names = set()
        self.rag.llm = MagicMock()
        mock_chain = MagicMock()
        # 先回傳格式錯誤（會觸發 JSONDecodeError），再回傳正確 JSON
        mock_chain.invoke.side_effect = [
            '{name: 無引號, price: 1000}',  # 無法解析的 JSON
            '{"name": "F", "price": 7000, "area": 70, "features": "大", "style": "現代", "maxOccupancy": "7"}'
        ]
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertEqual(result, {
            "name": "F", "price": 7000, "area": 70, "features": "大", "style": "現代", "maxOccupancy": 7
        })

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_auto_recommend_room_no_json_match(self, mock_prompt):
        # LLM 回傳內容中沒有任何大括號，match 會是 None，應觸發 continue
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = []
        self.rag.used_names = set()
        self.rag.llm = MagicMock()
        mock_chain = MagicMock()
        # 先回傳沒有大括號的內容，再回傳正確 JSON
        mock_chain.invoke.side_effect = [
            '這裡沒有json格式',
            '{"name": "G", "price": 8000, "area": 80, "features": "大", "style": "現代", "maxOccupancy": "8"}'
        ]
        mock_prompt.return_value.__or__.return_value = mock_chain
        result = self.rag.auto_recommend_room()
        self.assertEqual(result, {
            "name": "G", "price": 8000, "area": 80, "features": "大", "style": "現代", "maxOccupancy": 8
        })

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_review_recommendation_fully_match(self, mock_prompt):
        """LLM 回覆完全符合需求"""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '推薦內容符合使用者需求，無需變更。'
        mock_prompt.return_value.__or__.return_value = mock_chain
        user_question = '我要兩人房，現代風格'
        llm_output = '推薦內容...'
        result = self.rag.review_recommendation(user_question, llm_output)
        self.assertEqual(result, '推薦內容符合使用者需求，無需變更。')

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_review_recommendation_not_fully_match(self, mock_prompt):
        """LLM 回覆不完全符合需求，需重新推薦"""
        reply = '目前沒有完全符合的房型，以下是最接近的建議\n房型A\n推薦理由...'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = reply
        mock_prompt.return_value.__or__.return_value = mock_chain
        user_question = '我要三人房，工業風格'
        llm_output = '推薦內容...'
        result = self.rag.review_recommendation(user_question, llm_output)
        self.assertEqual(result, reply)

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_review_recommendation_unexpected(self, mock_prompt):
        """LLM 回覆未預期內容"""
        reply = '這是一個未預期的回覆'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = reply
        mock_prompt.return_value.__or__.return_value = mock_chain
        user_question = '我要四人房，日式風格'
        llm_output = '推薦內容...'
        result = self.rag.review_recommendation(user_question, llm_output)
        self.assertEqual(result, reply)

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_llm_prediction_normal(self, mock_prompt):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "推薦房型：\n房型名稱\n推薦理由：..."
        mock_prompt.return_value.__or__.return_value = mock_chain

        question = "我要三人房"
        rooms_summary = "名稱:房A 價格:2000 面積:20 特色:大 風格:工業風 床數:2"
        result = self.rag.LLM_Prediction(question, rooms_summary)

        self.assertEqual(result, "推薦房型：\n房型名稱\n推薦理由：...")
        mock_prompt.assert_called_once()
        mock_chain.invoke.assert_called_once_with({"input": question, "rooms": rooms_summary})

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_llm_prediction_empty_rooms(self, mock_prompt):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "目前沒有符合條件的房型"
        mock_prompt.return_value.__or__.return_value = mock_chain

        question = "我要日式風格"
        rooms_summary = ""
        result = self.rag.LLM_Prediction(question, rooms_summary)

        self.assertEqual(result, "目前沒有符合條件的房型")
        mock_chain.invoke.assert_called_once_with({"input": question, "rooms": rooms_summary})

    @patch('src.RAG.ChatPromptTemplate.from_messages')
    def test_llm_prediction_special_characters(self, mock_prompt):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "推薦房型：\n房型名稱：豪華房\n推薦理由：適合您"
        mock_prompt.return_value.__or__.return_value = mock_chain

        question = "我要$3000~$5000的房型"
        rooms_summary = "名稱:豪華房 價格:4000 面積:30 特色:大 風格:現代風 床數:2"
        result = self.rag.LLM_Prediction(question, rooms_summary)

        self.assertIn("豪華房", result)
        mock_chain.invoke.assert_called_once_with({"input": question, "rooms": rooms_summary})