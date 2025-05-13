import unittest
from RAG import RAGPipeline



class RAT_TEST(unittest.TestCase):
    def setUp(self):
        self.rag = RAGPipeline(rf'F:\GITHUB\Room-Type-Recommendation-System\static\rooms.json')

    def test_Prompt(self):
        print(self.rag.query("嗨，你好"))
        print(self.rag.query("請問有適合三個人入住，預算3000元內的房型嗎？"))


if __name__ == '__main__':
    unittest.main()
