import unittest
from RAG import RAGPipeline


class RAT_TEST(unittest.TestCase):
    def setUp(self):
        self.Rag = RAGPipeline(rf'F:\GITHUB\Room-Type-Recommendation-System\static\rooms.json')


    def testPromptInput(self):
        question = """
        我想要有日式建築的房型，請問在我的 rooms.json 資料中是否有符合此特色的房型?
        若有，請回傳「房型名稱 (價格:)」
        """

        print(self.Rag.query(question))




if __name__ == '__main__':
    unittest.main()
