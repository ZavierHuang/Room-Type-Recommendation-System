import json
import os
import pathlib
import unittest

from bs4 import BeautifulSoup

from app import app, ROOT


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.ROOT = pathlib.Path(__file__).resolve().parent.parent
        app.config['TESTING'] = True

        with open(os.path.join(self.ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as jsonFile:
            self.jsonData = json.load(jsonFile)

        self.client = app.test_client()


    def tearDown(self):
        with open(os.path.join(self.ROOT, 'static/rooms.json'), 'w', encoding='utf-8') as jsonFile:
            jsonFile.write(json.dumps(self.jsonData, indent=4, ensure_ascii=False))

    # 測試首頁是否能正確回應
    def test_index(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        text = resp.data.decode('utf-8')
        self.assertIn('<title>房型推薦系統</title>', text)

    # 測試聊天功能是否能正確回應
    def test_chat(self):
        resp = self.client.post('/chat', json={'message': '請推薦房型'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('response', resp.get_json())

    # 測試登入成功的情況
    def test_login_success(self):
        resp = self.client.post('/login', data={'username': 'admin', 'password': 'admin'}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        soup = BeautifulSoup(resp.data, 'html.parser')

        # 找出 class 為 'btn btn-success' 並且觸發 modal 的 button
        button = soup.find('button', {
            'class': 'btn btn-success',
            'data-bs-toggle': 'modal',
            'data-bs-target': '#addRoomModal'
        })

        self.assertIsNotNone(button)  # 確認這個按鈕存在
        
    # 測試登入失敗的情況
    def test_login_fail(self):
        resp = self.client.post('/login', data={'username': 'wrong', 'password': 'wrong'})
        # 斷言時將 bytes 轉為 str 再比對，避免非 ASCII 字元造成 SyntaxError
        self.assertIn('帳號或密碼錯誤', resp.data.decode('utf-8'))

    # 測試登出功能是否能正確回應
    def test_logout(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
        resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        text = resp.data.decode('utf-8')
        self.assertIn('<title>房型推薦系統</title>', text)

    # 測試未登入時存取後端頁面是否會被重定向到登入頁面
    def test_backend_requires_login(self):
        resp = self.client.get('/backend', follow_redirects=True)
        text = resp.data.decode('utf-8')
        self.assertIn('<title>登入頁面</title>', text)

    # 測試登入後是否能正確存取後端頁面
    def test_backend_with_login(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
        resp = self.client.get('/backend')
        self.assertEqual(resp.status_code, 200)

        soup = BeautifulSoup(resp.data, 'html.parser')

        # 找出 class 為 'btn btn-success' 並且觸發 modal 的 button
        button = soup.find('button', {
            'class': 'btn btn-success',
            'data-bs-toggle': 'modal',
            'data-bs-target': '#addRoomModal'
        })

        self.assertIsNotNone(button)  # 確認這個按鈕存在

    # 測試自動推薦房型功能是否能正確回應
    def test_auto_recommend(self):
        resp = self.client.get('/auto_recommend')
        self.assertIn(resp.status_code, (200, 404))

    # 測試自動推薦房型功能在失敗情況下的回應
    def test_auto_recommend_fail(self):
        import app
        original_func = app.rag.auto_recommend_room
        app.rag.auto_recommend_room = lambda: None
        resp = self.client.get('/auto_recommend')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json().get('error'), '無法推薦房型')
        app.rag.auto_recommend_room = original_func

    # 測試生成房型圖片功能是否能正確回應
    def test_generate_room_image(self):
        data = {'name': '測試房型', 'features': '測試'}
        resp = self.client.post('/generate_room_image', json=data)
        self.assertIn(resp.status_code, (200, 500))

    # 測試生成房型圖片功能在成功情況下的回應
    def test_generate_room_image_success(self):
        # 模擬 Text2Image.textToImage 回傳 True
        from unittest.mock import patch
        data = {'name': '測試房型', 'features': '測試'}
        with patch('src.Text2Image.Text2Image.textToImage', return_value=True):
            resp = self.client.post('/generate_room_image', json=data)
            self.assertEqual(resp.status_code, 200)
            result = resp.get_json()
            self.assertIn('image_url', result)
            self.assertIn('static/image/img_', result['image_url'])

    # 測試新增房型功能是否能正確回應
    def test_add_room(self):
        # 建立一個暫時檔案模擬圖片
        image_path = os.path.join(ROOT, 'static/image/img_99_temp.png')
        with open(image_path, 'wb') as f:
            f.write(b'test')  # 假裝圖片內容

        data = {
            'name': '測試房型',
            'price': '1000',
            'area': '10坪',
            'features': '測試',
            'style': '現代',
            'maxOccupancy': '2',
            'image': '/static/image/img_99_temp.png'
        }
        resp = self.client.post('/add_room', json=data)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('success'))

        # 清理
        os.remove(os.path.join(ROOT, 'static/image/img_99.png'))

    # 測試新增房型功能在失敗情況下的回應
    def test_add_room_fail(self):
        # 傳入不存在的圖片路徑，應觸發 return jsonify({'failure': False})
        data = {
            'name': '測試房型',
            'price': '1000',
            'area': '10坪',
            'features': '測試',
            'style': '現代',
            'maxOccupancy': '2',
            'image': 'not_a_valid_path.png'
        }
        resp = self.client.post('/add_room', json=data)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get('failure'), False)

    # 測試新增房型時臨時圖片是否被正確重命名
    def test_add_room_temp_image_rename(self):
        ROOT = pathlib.Path(__file__).resolve().parent.parent
        # 準備一個假的 temp 檔案
        temp_img_path = os.path.join(ROOT, 'static/image/img_99_temp.png')
        with open(temp_img_path, 'wb') as f:
            f.write(b'fake image data')
        data = {
            'name': '測試房型_temp',
            'price': '1000',
            'area': '10',
            'features': '測試',
            'style': '現代',
            'maxOccupancy': '2',
            'image': '../static/image/img_99_temp.png'
        }
        resp = self.client.post('/add_room', json=data)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('success'))
        # 檢查 temp 檔已被更名
        self.assertFalse(os.path.exists(temp_img_path))
        self.assertTrue(os.path.exists(os.path.join(ROOT, 'static/image/img_99.png')))
        os.remove(os.path.join(ROOT, 'static/image/img_99.png'))

    # 測試新增房型時臨時圖片不存在的情況
    def test_add_room_temp_image_not_exist(self):
        data = {
            'name': '測試房型_temp2',
            'price': '1000',
            'area': '10',
            'features': '測試',
            'style': '現代',
            'maxOccupancy': '2',
            'image': '../static/image/img_100_temp.png'
        }
        resp = self.client.post('/add_room', json=data)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get('failure'), False)

