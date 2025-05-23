import os
import pathlib

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from src.RAG import RAGPipeline
from src.Text2Image import Text2Image
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 新增 session 密鑰
ROOT = pathlib.Path(__file__).resolve().parent

rag = RAGPipeline(os.path.join(ROOT, 'static/rooms.json'))

@app.route('/')
def index():
    # 讀取房型資料 JSON
    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    return render_template('index.html', rooms=rooms)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = rag.query(user_input)
    return jsonify({'response': response})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('blank'))
        else:
            return render_template('login.html', error='帳號或密碼錯誤')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/blank')
def blank():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    # 讀取房型資料 JSON
    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    return render_template('blank.html', rooms=rooms)

@app.route('/auto_recommend', methods=['GET'])
def auto_recommend():
    room = rag.auto_recommend_room()
    if room:
        return jsonify(room)
    else:
        return jsonify({'error': '無法推薦房型'}), 404

@app.route('/generate_room_image', methods=['POST'])
def generate_room_image():
    data = request.get_json()

    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    # 產生新 id
    new_id = max([room['id'] for room in rooms], default=-1) + 1

    image_filename = f"img_{new_id}.png"
    image_path = os.path.join(ROOT, f"static/image/{image_filename}")
    t2i = Text2Image(data, image_path)
    success = t2i.textToImage()
    if not success:
        return jsonify({'error': '圖片生成失敗'}), 500
    return jsonify({'image_url': f'/static/image/{image_filename}'})

@app.route('/add_room', methods=['POST'])
def add_room():
    data = request.get_json()
    # 讀取現有資料
    with open(os.path.join(ROOT, 'static/rooms.json'), 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    # 產生新 id
    new_id = max([room['id'] for room in rooms], default=-1) + 1
    # 處理圖片路徑（如果是完整 URL 只取最後檔名）
    image_url = data.get('image', '')
    if image_url and '/static/image/' in image_url:
        image_filename = image_url.split('/static/image/')[-1].split('?')[0]  # 去除快取參數
        image = os.path.join(ROOT, 'static', 'image', image_filename)
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
        'image': f'../static/image/{image_filename}' if image else '',
    }
    rooms.append(new_room)
    # 寫回 JSON
    with open(os.path.join(ROOT, 'static/rooms.json'), 'w', encoding='utf-8') as f:
        json.dump(rooms, f, ensure_ascii=False, indent=4)
    # 新增：同步 RAG 內部資料
    rag.data = rooms
    rag.maxID = max([int(item['id']) for item in rag.data])
    rag.docs = [
        __import__('langchain').docstore.document.Document(
            page_content=f"名稱:{item['name']} 價格:{item['price']} 面積:{item['area']} 特色:{item['features']} 風格:{item.get('style', '')} 床數:{item.get('maxOccupancy', '')}"
        ) for item in rag.data
    ]
    rag.vectorstore = __import__('langchain_community').vectorstores.Chroma.from_documents(
        rag.docs, __import__('langchain_community').embeddings.FastEmbedEmbeddings()
    )
    rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 10})
    return jsonify({'success': True})

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True
    app.run(debug=True)

# Ctrl + ALT + B = Github Copilot