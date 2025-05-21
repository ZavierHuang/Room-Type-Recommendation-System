import unittest
import os
from imageAI.Text2Image import Text2Image
import re
import pathlib
import unittest

class TextToImageTest(unittest.TestCase):
    def normalize(self, text):
        # Replace all symbols other than numbers and letters with empty
        return re.sub(r'[^A-Za-z0-9]', '', text)

    def setUp(self):
        self.ROOT = pathlib.Path(__file__).parent.parent
        jsonData = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        imageFilePath = 'test/test.png'
        self.text2Image = Text2Image(jsonData, imageFilePath)

    def test_convert_to_Sentence_From_Json(self):
        prompt = self.text2Image.convertToSentenceFromJson()
        
        expected = """
        一張高品質的和式套房室內照片，30平方米，包含日式榻榻米、茶几、浴衣、日式拉門，2人房，日式風格.
        """
        self.assertEqual(self.normalize(prompt), self.normalize(expected))
    
    def test_translator_api(self):
        prompt = """
        你好，世界
        """
        self.text2Image.setPrompt(prompt)
        expected = """
        Hello World.
        """
        result = self.text2Image.TranslatorAPI()
        self.assertEqual(self.normalize(result), self.normalize(expected))
        
    def test_text_to_image_integration(self):
        self.assertTrue(self.text2Image.textToImage())
        self.assertTrue(os.path.exists(self.text2Image.getImageFilePath()))

    def test_set_and_get_prompt(self):
        self.text2Image.setPrompt('測試用prompt')
        self.assertEqual(self.text2Image.getPrompt(), '測試用prompt')

    def test_set_and_get_json_data(self):
        new_json = {
            "name": "現代雙人房",
            "price": "4000",
            "area": "25",
            "features": "現代設計、落地窗",
            "style": "現代",
            "maxOccupancy": "2人房"
        }
        self.text2Image.setJsonData(new_json)
        self.assertEqual(self.text2Image.getJsonData(), new_json)

    def test_set_and_get_image_file_path(self):
        new_path = 'static/image/test2.png'
        self.text2Image.setImageFilePath(new_path)
        self.assertTrue(self.text2Image.getImageFilePath().endswith('static/image/test2.png'))

    def test_generate_image_mock(self):
        # 測試 generateImage 方法是否能被呼叫（不實際產生圖檔）
        self.text2Image.prompt = 'A modern room.'
        class DummyPipe:
            def __call__(self, prompt):
                class DummyResult:
                    images = [type('img', (), {'resize': lambda self, s: self, 'show': lambda self: None, 'save': lambda self, p: None})()]
                return DummyResult()
        self.text2Image.pipe = DummyPipe()
        try:
            self.text2Image.generateImage()
        except Exception as e:
            self.fail(f"generateImage raised Exception unexpectedly: {e}")



if __name__ == '__main__':
    unittest.main()
