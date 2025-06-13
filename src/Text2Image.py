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

        # 初始化 Stable Diffusion 模型和 scheduler
        self.model_id = None
        self.scheduler = None
        self.pipe = None



    def setJsonData(self, jsonData):
        self.jsonData = jsonData

    """
    使用 Google 翻譯 API 將中文翻譯成英文
    """
    def TranslatorAPI(self):
        try:
            result = GoogleTranslator(source='zh-CN', target='en').translate(self.prompt)
            return result
        except Exception as e:
            print(e)
            return None

    """
    將 JSON 房型敘述轉換成文字敘述
    """
    def convertToSentenceFromJson(self):
        if not self.jsonData:
            return None

        if (not self.jsonData.get("name") or
                not self.jsonData.get("area") or
                not self.jsonData.get("features") or
                not self.jsonData.get("style") or
                not self.jsonData.get("maxOccupancy")):
            return None

        prompt = f"""一張高品質的{self.jsonData['name']}室內照片，{self.jsonData['style']}風格，面積{self.jsonData['area']}平方米，主要特色包含{self.jsonData['features']}，{self.jsonData['maxOccupancy']}。"""

        return prompt

    """
    此方法執行以下步驟：
    1. 使用 StableDiffusionPipeline 並根據 self.prompt 提供的文本提示生成圖像。
    2. 將生成的圖像調整為指定大小 (1000x512)。
    3. 將調整後的圖像保存到指定的文件路徑。

    return
    - 如果圖像生成成功，返回 True。
    - 如果過程中出現錯誤，返回 False。

    """
    def generateImage(self):
        try:
            image = self.pipe(self.prompt).images[0]
            new_size = (1000, 512)
            resized_image = image.resize(new_size)
            resized_image.save(self.imageFilePath)
            return True
        except Exception as e:
            print(f"Error generating image: {e}")
            return False

    """
    將 JSON 數據轉換為圖像的完整流程。

    此方法執行以下步驟：
    1. 將 JSON 數據轉換為中文描述句子。
    2. 使用 Google 翻譯 API 將中文描述翻譯為英文。
    3. 使用 Stable Diffusion Pipeline 根據翻譯後的描述生成圖像。
    4. 將生成的圖像調整大小並儲存到指定路徑。

    return
    - 如果圖像生成成功，返回 True。
    - 如果過程中出現錯誤，返回 False。

    """
    def textToImage(self):
        self.model_id = "stabilityai/stable-diffusion-2"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        #  self.pipe 負責整合模型與流程 → 從文字生成圖片
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, scheduler=self.scheduler,
                                                            torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

        self.prompt = self.convertToSentenceFromJson()

        if self.prompt is None:
            return False

        self.prompt = self.TranslatorAPI()
        return self.generateImage() if self.prompt is not None else False
