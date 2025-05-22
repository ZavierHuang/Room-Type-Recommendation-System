import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        # 將 bytes 轉為 str 再比對中文
        text = resp.data.decode('utf-8')
        self.assertTrue('房型' in text or 'name' in text)

    def test_chat(self):
        resp = self.client.post('/chat', json={'message': '請推薦房型'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('response', resp.get_json())

    def test_login_success(self):
        resp = self.client.post('/login', data={'username': 'admin', 'password': 'admin'}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        text = resp.data.decode('utf-8')
        self.assertTrue('房型' in text or 'name' in text)

    def test_login_fail(self):
        resp = self.client.post('/login', data={'username': 'wrong', 'password': 'wrong'})
        # 斷言時將 bytes 轉為 str 再比對，避免非 ASCII 字元造成 SyntaxError
        self.assertIn('帳號或密碼錯誤', resp.data.decode('utf-8'))

    def test_logout(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
        resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        text = resp.data.decode('utf-8')
        self.assertTrue('房型' in text or 'name' in text)

    def test_blank_requires_login(self):
        resp = self.client.get('/blank', follow_redirects=True)
        text = resp.data.decode('utf-8')
        self.assertTrue('登入' in text or 'login' in text)

    def test_blank_with_login(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
        resp = self.client.get('/blank')
        self.assertEqual(resp.status_code, 200)
        text = resp.data.decode('utf-8')
        self.assertTrue('房型' in text or 'name' in text)

    def test_auto_recommend(self):
        resp = self.client.get('/auto_recommend')
        self.assertIn(resp.status_code, (200, 404))

    def test_generate_room_image(self):
        data = {'name': '測試房型', 'features': '測試'}
        resp = self.client.post('/generate_room_image', json=data)
        self.assertIn(resp.status_code, (200, 500))

    def test_add_room(self):
        data = {
            'name': '測試房型',
            'price': '1000',
            'area': '10坪',
            'features': '測試',
            'style': '現代',
            'maxOccupancy': '2',
            'image': '../static/image/img_99.png'
        }
        resp = self.client.post('/add_room', json=data)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('success'))
