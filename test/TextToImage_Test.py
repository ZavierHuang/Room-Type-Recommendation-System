import unittest
import os
from Configure import ROOT
from imageAI.Text2Image import Text2Image
import re

class TextToImageTest(unittest.TestCase):
    def normalize(self, text):
        # Replace all symbols other than numbers and letters with empty
        return re.sub(r'[^A-Za-z0-9]', '', text)

    def setUp(self):
        jsonData = {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        imageFilePath = os.path.join(ROOT,'test/test.png')
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
        self.text2Image.textToImage()
        self.assertTrue(os.path.exists(self.text2Image.getImageFilePath()))

if __name__ == '__main__':
    unittest.main()
