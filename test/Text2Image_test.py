import unittest
from unittest.mock import patch, MagicMock
from src.Text2Image import Text2Image

class TestText2ImageTranslatorAPI(unittest.TestCase):
    def setUp(self):
        self.text2image = Text2Image({}, "test.png")

    """
    TranslatorAPI 測試
    """
    def test_translator_api_success(self):
        self.text2image.prompt = "這是一個測試"
        result = self.text2image.TranslatorAPI()
        self.assertEqual(result, "This is a test")

    def test_translator_api_prompt_none(self):
        self.text2image.prompt = None
        result = self.text2image.TranslatorAPI()
        self.assertIsNone(result)

    """
    JSON 與 Sentence 之間的轉換測試
    """
    def test_convert_to_sentence_from_json_normal(self):
        json_data = {
            "name": "和式套房",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        prompt = self.text2image.convertToSentenceFromJson()
        expected = "一張高品質的和式套房室內照片，日式風格，面積30平方米，主要特色包含日式榻榻米、茶几、浴衣、日式拉門，2人房。"
        self.assertEqual(prompt, expected)

    def test_convert_to_sentence_from_json_part_json(self):
        json_data = {
            "name": "現代雙人房",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        prompt = self.text2image.convertToSentenceFromJson()
        self.assertIsNone(prompt)

    def test_convert_to_sentence_from_json_None_json(self):
        self.text2image.setJsonData(None)
        prompt = self.text2image.convertToSentenceFromJson()
        self.assertIsNone(prompt)


    """
    測試 generateImage 方法在正常流程下能成功產生圖片，並做以下動作：
    1. 使用預設的 prompt 呼叫生成圖片的 pipeline（self.text2image.pipe）
    2. 對生成的圖片做 resize (改變大小)
    3. 把 resize 後的圖片存檔到指定路徑 (self.text2image.imageFilePath)
    4. 最後返回 True (表示成功)
    """
    def test_generate_image_success(self):
        self.text2image.prompt = "test prompt"

        #  mock 物件來模擬生成的圖片
        #  mock 物件來模擬 resize 後的圖片
        mock_image = MagicMock()
        mock_resized_image = MagicMock()
        mock_image.resize.return_value = mock_resized_image
        mock_pipe = MagicMock(return_value=MagicMock(images=[mock_image]))

        # 模擬 save() 方法
        mock_resized_image.save = MagicMock()

        self.text2image.pipe = mock_pipe

        result = self.text2image.generateImage()
        self.assertTrue(result)

        # 驗證 Mock 呼叫是否正確
        # 確認 pipe 只呼叫一次，參數是 self.text2image.prompt
        # 確認 resize() 方法有被呼叫，且參數為 (1000, 512)
        # 確認 .save() 有被呼叫，且檔案路徑正
        mock_pipe.assert_called_once_with(self.text2image.prompt)
        mock_image.resize.assert_called_once_with((1000, 512))
        mock_resized_image.save.assert_called_once_with(self.text2image.imageFilePath)

    def test_generate_image_exception(self):
        self.text2image.prompt = "test prompt"

        # side_effect 是 unittest.mock.MagicMock() 的一個特性 → 模擬異常情境
        mock_pipe = MagicMock(side_effect=Exception("fail"))
        self.text2image.pipe = mock_pipe
        result = self.text2image.generateImage()
        self.assertFalse(result)

    """
    測試  textToImage() 方法能否成功完成整個流程
        1.初始化 Scheduler 與 Pipeline
        2.呼叫 convertToSentenceFromJson() 轉換 JSON 資料為文字描述
        3.呼叫 TranslatorAPI() 翻譯文字描述為英文
        4.呼叫 generateImage() 產生圖片
        5.最後回傳成功 (True)
        
    @patch Mock 外部依賴套件
    """
    @patch('src.Text2Image.StableDiffusionPipeline')
    @patch('src.Text2Image.EulerDiscreteScheduler')
    @patch('src.Text2Image.GoogleTranslator')
    def test_text_to_image_success(self, mock_translator, mock_scheduler, mock_pipeline):
        json_data = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)

        # 模擬 Scheduler 載入
        mock_scheduler.from_pretrained.return_value = 'mock_scheduler'

        # 模擬 Pipeline 載入與生成圖片
        mock_pipe_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        # 模擬 Pipeline 的 to() 方法(to() 是將模型移動到指定設備（如 GPU），這裡模擬直接回傳自己
        mock_pipe_instance.to.return_value = mock_pipe_instance

        # textToImage() 內部應該會呼叫 generateImage() → 這裡直接 Mock 成回傳 True
        mock_translator.return_value.translate.return_value = 'translated prompt'
        with patch.object(self.text2image, 'generateImage', return_value=True) as mock_gen_img:
            result = self.text2image.textToImage()
            self.assertTrue(result)
            # 確認 generateImage() 被呼叫一次
            mock_gen_img.assert_called_once()

    """
    Step2. convertToSentenceFromJson 執行失敗，textToImage() 應該回傳 False
    """
    @patch('src.Text2Image.StableDiffusionPipeline')
    @patch('src.Text2Image.EulerDiscreteScheduler')
    @patch('src.Text2Image.GoogleTranslator')
    def test_text_to_image_prompt_none(self, mock_translator, mock_scheduler, mock_pipeline):
        self.text2image.setJsonData(None)
        with patch.object(self.text2image, 'convertToSentenceFromJson', return_value=None):
            result = self.text2image.textToImage()
            self.assertFalse(result)

    """
    Step4. translatorAPI 執行失敗，textToImage() 應該回傳 False
    """
    @patch('src.Text2Image.StableDiffusionPipeline')
    @patch('src.Text2Image.EulerDiscreteScheduler')
    @patch('src.Text2Image.GoogleTranslator')
    def test_text_to_image_translate_fail(self, mock_translator, mock_scheduler, mock_pipeline):
        # 翻譯失敗
        json_data = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        mock_scheduler.from_pretrained.return_value = 'mock_scheduler'
        mock_pipe_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_translator.return_value.translate.side_effect = Exception('fail')
        with patch.object(self.text2image, 'generateImage', return_value=True):
            result = self.text2image.textToImage()
            self.assertFalse(result)

    """
    Step4. generateImage 執行失敗，textToImage() 應該回傳 False
    """
    @patch('src.Text2Image.StableDiffusionPipeline')
    @patch('src.Text2Image.EulerDiscreteScheduler')
    @patch('src.Text2Image.GoogleTranslator')
    def test_text_to_image_generate_image_fail(self, mock_translator, mock_scheduler, mock_pipeline):
        # 生成圖片失敗
        json_data = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        mock_scheduler.from_pretrained.return_value = 'mock_scheduler'
        mock_pipe_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_translator.return_value.translate.return_value = 'translated prompt'
        with patch.object(self.text2image, 'generateImage', return_value=False):
            result = self.text2image.textToImage()
            self.assertFalse(result)
