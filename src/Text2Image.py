import os

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from deep_translator import GoogleTranslator
import pathlib


class Text2Image:
    def __init__(self, jsonData, imageFilePath):
        self.ROOT = pathlib.Path(__file__).resolve().parent.parent
        self.jsonData = jsonData
        self.prompt = None
        self.imageFilePath = imageFilePath

    def setJsonData(self, jsonData):
        self.jsonData = jsonData

    def TranslatorAPI(self):
        try:
            result = GoogleTranslator(source='zh-CN', target='en').translate(self.prompt)
            return result
        except Exception as e:
            print(e)
            return None

    def convertToSentenceFromJson(self):
        """
        self.jsonData
        {
            "name": "和式套房",
            "price": "5000",
            "area": "30",
            "features": "日式榻榻米、茶几、浴衣、日式拉門",
            "style": "日式",
            "maxOccupancy": "2人房"
        }
        """
        if not self.jsonData:
            return None

        if (not self.jsonData.get("name") or
                not self.jsonData.get("area") or
                not self.jsonData.get("features") or
                not self.jsonData.get("style") or
                not self.jsonData.get("maxOccupancy")):
            return None

        prompt = f"""一張高品質的{self.jsonData['name']}室內照片，{self.jsonData['area']}平方米，包含{self.jsonData['features']}，{self.jsonData['maxOccupancy']}，{self.jsonData['style']}風格。"""

        return prompt
    
    def generateImage(self):
        try:
            image = self.pipe(self.prompt).images[0]
            new_size = (1000, 512)
            resized_image = image.resize(new_size)
            # resized_image.show()
            resized_image.save(self.imageFilePath)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

    def textToImage(self):
        self.model_id = "stabilityai/stable-diffusion-2"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, scheduler=self.scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

        self.prompt = self.convertToSentenceFromJson()

        if self.prompt:
            self.prompt = self.TranslatorAPI()
            return self.generateImage() if self.prompt is not None else False

        return False
