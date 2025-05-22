import unittest
from unittest.mock import patch, MagicMock
from src.Text2Image import Text2Image

class TestText2ImageTranslatorAPI(unittest.TestCase):
    def setUp(self):
        self.text2image = Text2Image({}, "test.png")

    def test_translator_api_success(self):
        self.text2image.prompt = "這是一個測試"
        result = self.text2image.TranslatorAPI()
        self.assertEqual(result, "This is a test")

    def test_translator_api_prompt_none(self):
        self.text2image.prompt = None
        result = self.text2image.TranslatorAPI()
        self.assertIsNone(result)

    def test_convert_to_sentence_from_json_normal(self):
        json_data = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        prompt = self.text2image.convertToSentenceFromJson()
        expected = "一張高品質的和式套房室內照片，30平方米，包含日式榻榻米、茶几、浴衣、日式拉門，2人房，日式風格。"
        self.assertEqual(prompt, expected)

    def test_convert_to_sentence_from_json_missing_fields(self):
        json_data = {
            "name": "現代雙人房",
            "area": "25",
            "features": "大床、落地窗",
            "style": "現代",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        prompt = self.text2image.convertToSentenceFromJson()
        expected = "一張高品質的現代雙人房室內照片，25平方米，包含大床、落地窗，2人房，現代風格。"
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

    def test_generate_image_success(self):
        self.text2image.prompt = "test prompt"
        mock_image = MagicMock()
        mock_resized_image = MagicMock()
        mock_image.resize.return_value = mock_resized_image
        mock_pipe = MagicMock(return_value=MagicMock(images=[mock_image]))
        mock_resized_image.save = MagicMock()
        self.text2image.pipe = mock_pipe

        result = self.text2image.generateImage()
        self.assertTrue(result)
        mock_pipe.assert_called_once_with(self.text2image.prompt)
        mock_image.resize.assert_called_once_with((1000, 512))
        mock_resized_image.save.assert_called_once_with(self.text2image.imageFilePath)

    def test_generate_image_exception(self):
        self.text2image.prompt = "test prompt"
        mock_pipe = MagicMock(side_effect=Exception("fail"))
        self.text2image.pipe = mock_pipe
        result = self.text2image.generateImage()
        self.assertFalse(result)

    @patch('src.Text2Image.StableDiffusionPipeline')
    @patch('src.Text2Image.EulerDiscreteScheduler')
    @patch('src.Text2Image.GoogleTranslator')
    def test_text_to_image_success(self, mock_translator, mock_scheduler, mock_pipeline):
        # Mock prompt組合
        json_data = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        self.text2image.setJsonData(json_data)
        # Mock Scheduler
        mock_scheduler.from_pretrained.return_value = 'mock_scheduler'
        # Mock Pipeline
        mock_pipe_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        mock_pipe_instance.to.return_value = mock_pipe_instance
        # Mock Translator
        mock_translator.return_value.translate.return_value = 'translated prompt'
        # Mock generateImage
        with patch.object(self.text2image, 'generateImage', return_value=True) as mock_gen_img:
            result = self.text2image.textToImage()
            self.assertTrue(result)
            mock_gen_img.assert_called_once()

    @patch('src.Text2Image.StableDiffusionPipeline')
    @patch('src.Text2Image.EulerDiscreteScheduler')
    @patch('src.Text2Image.GoogleTranslator')
    def test_text_to_image_prompt_none(self, mock_translator, mock_scheduler, mock_pipeline):
        # prompt 組合失敗
        self.text2image.setJsonData(None)
        with patch.object(self.text2image, 'convertToSentenceFromJson', return_value=None):
            result = self.text2image.textToImage()
            self.assertFalse(result)

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
