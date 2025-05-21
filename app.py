from flask import Flask, render_template, request, jsonify, redirect, url_for
from src.RAG import RAGPipeline
import json

app = Flask(__name__)
rag = RAGPipeline('static/rooms.json')

@app.route('/')
def index():
    # 讀取房型資料 JSON
    with open('static/rooms.json', 'r', encoding='utf-8') as f:
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
            return redirect(url_for('blank'))
        else:
            # 登入失敗，回到登入頁並顯示錯誤訊息
            return render_template('login.html', error='帳號或密碼錯誤')
    return render_template('login.html')

@app.route('/blank')
def blank():
    with open('static/rooms.json', 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    return render_template('blank.html', rooms=rooms)

@app.route('/auto_recommend', methods=['GET'])
def auto_recommend():
    room = rag.auto_recommend_room()
    if room:
        return jsonify(room)
    else:
        return jsonify({'error': '無法推薦房型'}), 404

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True
    app.run(debug=True)

#.venv\Scripts\activate  
# Ctrl + ALT + B = Github Copilot