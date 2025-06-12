import unittest
from unittest.mock import patch, MagicMock
from src.RAG import RAGPipeline

class TestRAGPipelineQuery(unittest.TestCase):
    def setUp(self):
        self.rag = RAGPipeline.__new__(RAGPipeline)
        self.rag.data = [
            {"name": "A", "price": 1000, "area": 10, "features": "大", "style": "工業風", "maxOccupancy": 2},
            {"name": "B", "price": 2000, "area": 20, "features": "小", "style": "北歐風", "maxOccupancy": 3}
        ]

    @patch.object(RAGPipeline, 'classify_intent')
    def test_query_greeting(self, mock_intent):
        mock_intent.return_value = "打招呼"
        result = self.rag.query("你好")
        self.assertEqual(result["rooms"], [])
        self.assertIn("很高興為您服務", result["conclusion"])

    @patch.object(RAGPipeline, 'classify_intent')
    def test_query_other(self, mock_intent):
        mock_intent.return_value = "其他"
        result = self.rag.query("你是誰")
        self.assertEqual(result["rooms"], [])
        self.assertIn("只提供房型相關的建議", result["conclusion"])

    """
    測試當使用者詢問房型推薦（如："我要1000~2000元的房型"）且推薦內容完全符合需求時，整個流程是否會：
    1. 正確分類意圖為「房型推薦」    
    2. 正確擷取並過濾價格與面積範圍
    3. 正確呼叫 LLM 進行推薦
    4. 得到「符合需求」的審核結論
    5. 回傳完整推薦內容與房型資訊
    """
    @patch.object(RAGPipeline, 'classify_intent')
    @patch.object(RAGPipeline, 'getRoomSummaryByRAG')
    @patch.object(RAGPipeline, 'extract_price_range')
    @patch.object(RAGPipeline, 'filter_by_price_range')
    @patch.object(RAGPipeline, 'extract_area_range')
    @patch.object(RAGPipeline, 'filter_by_area_range')
    @patch.object(RAGPipeline, 'LLM_Prediction')
    @patch.object(RAGPipeline, 'review_recommendation')
    @patch.object(RAGPipeline, 'remove_duplicate_room_names')
    def test_query_room_recommend_fully_match(
        self, mock_remove_dup, mock_review, mock_llm, mock_filter_area, mock_extract_area,
        mock_filter_price, mock_extract_price, mock_get_summary, mock_intent
    ):
        mock_intent.return_value = "房型推薦"
        mock_get_summary.return_value = "名稱:A 價格:1000 面積:10\n名稱:B 價格:2000 面積:20"
        mock_extract_price.return_value = (1000, 2000, False, False)
        mock_filter_price.return_value = "名稱:A 價格:1000 面積:10\n名稱:B 價格:2000 面積:20"
        mock_extract_area.return_value = (10, 20, False, False)
        mock_filter_area.return_value = "名稱:A 價格:1000 面積:10\n名稱:B 價格:2000 面積:20"
        mock_llm.return_value = "房型名稱：A\n推薦理由：好\n房型名稱：B\n推薦理由：棒\n結語：歡迎入住"
        mock_review.return_value = "推薦內容符合使用者需求"
        mock_remove_dup.side_effect = lambda x: x

        result = self.rag.query("我要1000~2000元的房型")
        self.assertIn("A", result["rooms"])
        self.assertIn("B", result["rooms"])
        # 應驗證推薦內容本身
        self.assertEqual(result["conclusion"], mock_llm.return_value)

    @patch.object(RAGPipeline, 'classify_intent')
    @patch.object(RAGPipeline, 'getRoomSummaryByRAG')
    @patch.object(RAGPipeline, 'extract_price_range')
    @patch.object(RAGPipeline, 'filter_by_price_range')
    @patch.object(RAGPipeline, 'extract_area_range')
    @patch.object(RAGPipeline, 'filter_by_area_range')
    @patch.object(RAGPipeline, 'LLM_Prediction')
    @patch.object(RAGPipeline, 'review_recommendation')
    @patch.object(RAGPipeline, 'remove_duplicate_room_names')
    def test_query_room_recommend_not_fully_match(
        self, mock_remove_dup, mock_review, mock_llm, mock_filter_area, mock_extract_area,
        mock_filter_price, mock_extract_price, mock_get_summary, mock_intent
    ):
        """
        測試當使用者詢問房型推薦（如："我要3000~4000元的房型，30~40坪"）但資料庫無完全符合時，
        1. 正確分類意圖為「房型推薦」
        2. 正確擷取並過濾價格與面積範圍
        3. 正確呼叫 LLM 進行推薦
        4. 得到「不符合需求」的審核結論
        """
        mock_intent.return_value = "房型推薦"
        mock_get_summary.return_value = "名稱:A 價格:1000 面積:10\n名稱:B 價格:2000 面積:20"
        mock_extract_price.return_value = (3000, 4000, False, False)
        mock_filter_price.return_value = ""  # 無符合價格範圍
        mock_extract_area.return_value = (30, 40, False, False)
        mock_filter_area.return_value = ""  # 無符合面積範圍
        mock_llm.return_value = "目前沒有完全符合的房型"
        mock_review.return_value = "目前沒有完全符合的房型"
        mock_remove_dup.side_effect = lambda x: x

        result = self.rag.query("我要3000~4000元的房型，30~40坪")
        self.assertEqual(result["rooms"], {})
        self.assertIn("沒有完全符合的房型", result["conclusion"])
