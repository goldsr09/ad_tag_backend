from flask import Flask, request, jsonify
from flask_cors import CORS

import sqlite3

app = Flask(__name__)
CORS(app)
DB_FILE = 'tags.db'

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS tag_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_tag TEXT NOT NULL,
                transformed_tag TEXT NOT NULL,
                ad_server TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

@app.route('/save_tag', methods=['POST'])
def save_tag():
    data = request.json
    original_tag = data.get('original_tag')
    transformed_tag = data.get('transformed_tag')
    ad_server = data.get('ad_server', None)
    if not original_tag or not transformed_tag:
        return jsonify({'error': 'original_tag and transformed_tag required'}), 400
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            'INSERT INTO tag_pairs (original_tag, transformed_tag, ad_server) VALUES (?, ?, ?)',
            (original_tag, transformed_tag, ad_server)
        )
        conn.commit()
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
