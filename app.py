from flask import Flask, render_template, request, jsonify
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

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True
    app.run(debug=True)

#.venv\Scripts\activate  
# Ctrl + ALT + B = Github Copilot