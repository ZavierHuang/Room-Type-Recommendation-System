from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from deep_translator import GoogleTranslator




class Text2Image:
    def __init__(self, jsonData, imageFilePath):
        self.jsonData = jsonData
        self.prompt = None
        self.imageFilePath = imageFilePath

    def setPrompt(self, prompt):
        self.prompt = prompt

    def setImageFilePath(self, imageFilePath):
        self.imageFilePath = imageFilePath

    def getPrompt(self):
        return self.prompt

    def getImageFilePath(self):
        return self.imageFilePath

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
        prompt = f"""一張高品質的{self.jsonData['name']}室內照片，{self.jsonData['area']}平方米，包含{self.jsonData['features']}，{self.jsonData['maxOccupancy']}，{self.jsonData['style']}風格。"""

        return prompt
    
    def generateImage(self):
        image = self.pipe(self.prompt).images[0]
        image.show()
        image.save(self.imageFilePath)

    def textToImage(self):
        self.model_id = "stabilityai/stable-diffusion-2"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, scheduler=self.scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

        self.prompt = self.convertToSentenceFromJson()
        self.prompt = self.TranslatorAPI()

        self.generateImage()
        

