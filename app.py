from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    # 讀取房型資料 JSON
    with open('static/rooms.json', 'r', encoding='utf-8') as f:
        rooms = json.load(f)
    return render_template('index.html', rooms=rooms)

if __name__ == '__main__':
    app.run(debug=True)
