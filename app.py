import os
import pathlib
import secrets

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from src.RAG import RAGPipeline
from src.Text2Image import Text2Image
import json

app = Flask(__name__)
# 設定 session 加密金鑰，保護用戶 session 資料安全，防止被竄改
app.secret_key = secrets.token_hex(16)
ROOT = pathlib.Path(__file__).resolve().parent

rag = RAGPipeline(os.path.join(ROOT, 'static/rooms.json'))

"""
首頁：顯示所有房型資料
"""
@app.route('/')
def index():
    # 讀取房型資料 JSON
    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    rag.data = rooms  # 同步 RAGPipeline 內部資料
    return render_template('index.html', rooms=rooms)

"""
聊天 API，使用 RAGPipeline 回應用戶輸入

處理聊天訊息，回傳 RAGPipeline 的回應。
範例：
    POST /chat
    body: {"message": "請推薦一個適合三人入住的房型"}
回傳：{"response": "推薦房型資訊..."}
"""
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']  # 取得用戶訊息
    response = rag.query(user_input)      # 取得 RAG 回應
    return jsonify({'response': response})

"""
登入頁面，使用 session 管理登入狀態
GET（顯示登入頁面） 和 POST（提交登入表單）
"""
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # 簡單帳號密碼驗證
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True  # 設定登入狀態
            return redirect(url_for('backend'))
        else:
            return render_template('login.html', error='帳號或密碼錯誤')
    return render_template('login.html')

"""
登出，清除 session 並導向首頁
"""
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # 移除登入狀態
    return redirect(url_for('index'))

"""
管理後台頁面，僅限登入後存取
"""
@app.route('/backend')
def backend():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # 未登入導向登入頁

    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    return render_template('backend.html', rooms=rooms)

"""
自動推薦房型 API
"""
@app.route('/auto_recommend', methods=['GET'])
def auto_recommend():
    room = rag.auto_recommend_room()
    if room:
        return jsonify(room)
    else:
        return jsonify({'error': '無法推薦房型'}), 404

"""
根據房型描述生成圖片 API
return {"image_url": "/static/image/img_XX_temp.png"}
"""
@app.route('/generate_room_image', methods=['POST'])
def generate_room_image():
    data = request.get_json()

    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)

    # 產生新 id
    new_id = max([room['id'] for room in rooms], default=-1) + 1

    image_filename = f"img_{new_id}_temp.png"
    image_path = os.path.join(ROOT, f"static/image/{image_filename}")
    t2i = Text2Image(data, image_path)
    success = t2i.textToImage()

    if not success:
        # 「Internal Server Error」（內部伺服器錯誤）500
        # 表示伺服器在處理請求時發生了未預期到的問題，導致無法完成請求
        return jsonify({'error': '圖片生成失敗'}), 500

    return jsonify({'image_url': f'/static/image/{image_filename}'})

"""
新增房型資料 API
input：
    body: {name, price, area, features, style, maxOccupancy, image}
return：{"success": True} 或錯誤訊息。
"""
@app.route('/add_room', methods=['POST'])
def add_room():
    # 接收前端資料
    data = request.get_json()

    # 讀取現有資料
    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)

    # 產生新 id
    new_id = max([room['id'] for room in rooms], default=-1) + 1

    # 處理圖片路徑（如果是完整 URL 只取最後檔名）
    # http://localhost/static/image/img21_temp.png?cache=abc123
    image_url = data.get('image', '')
    if image_url and '/static/image/' in image_url:
        # image_filename = 'img21_temp.png'
        # image = ROOT / static / image / img21_temp.png
        image_filename = image_url.split('/static/image/')[-1].split('?')[0]
        image = os.path.join(ROOT, 'static', 'image', image_filename)

        # new_image_filename = 'img21.png'
        # new_image = ROOT / static / image / img21.png
        new_image_filename = image_filename.replace('_temp.png', '.png')
        new_image = os.path.join(ROOT, 'static', 'image', new_image_filename)

        try:
            os.rename(image, new_image)
        except Exception as e:
            return jsonify({'failure': False, 'error': str(e)})

    else:
        return jsonify({'failure': False})

    # 新房型資料
    new_room = {
        'id': new_id,
        'name': data.get('name', ''),
        'price': data.get('price', ''),
        'area': data.get('area', ''),
        'features': data.get('features', ''),
        'style': data.get('style', ''),
        'maxOccupancy': data.get('maxOccupancy', '') + '人房',
        'image': f'../static/image/{new_image_filename}',
    }
    rooms.append(new_room)

    # 寫回 JSON
    with open(os.path.join(ROOT, 'static/rooms.json'), 'w', encoding='utf-8') as f:
        json.dump(rooms, f, ensure_ascii=False, indent=4)

    # 把剛剛更新完的 rooms.json 完整同步到 RAG pipeline 的內部資料
    rag.data = rooms

    # 找出目前資料中最大 ID，用來後續新增房型時自動遞增 ID
    rag.maxID = max([int(item['id']) for item in rag.data])

    # 把每一個房型轉成 langchain 的 Document 格式
    rag.docs = [
        __import__('langchain').docstore.document.Document(
            page_content=f"名稱:{item['name']} 價格:{item['price']} 面積:{item['area']} 特色:{item['features']} 風格:{item.get('style', '')} 床數:{item.get('maxOccupancy', '')}"
        ) for item in rag.data
    ]

    # 建立向量資料庫
    rag.vectorstore = __import__('langchain_community').vectorstores.Chroma.from_documents(
        rag.docs, __import__('langchain_community').embeddings.FastEmbedEmbeddings()
    )

    # 設定檢索器，這樣 RAGPipeline 就可以使用向量資料庫進行檢索
    rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 10})

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
    # Ctrl + ALT + B = Github Copilot