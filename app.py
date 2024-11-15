from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import numpy as np
from board_generator import generate_random_queens, generate_regions

app = Flask(__name__)
CORS(app)

class GameState:
    def __init__(self, n=8):
        self.n = n
        self.queens = generate_random_queens(n)
        self.regions = generate_regions(self.queens, n)
        self.marks = np.zeros((n, n), dtype=int)  # 0: unmarked, 1: X, 2: queen
        
    def to_dict(self):
        return {
            'regions': self.regions.tolist(),
            'marks': self.marks.tolist(),
            'size': self.n
        }
        
    def toggle_mark(self, row, col):
        self.marks[row, col] = (self.marks[row, col] + 1) % 3
        return self.marks[row, col]

# Global game state (will be initialized when size is selected)
game = None

@app.route('/api/new_game/<int:size>')
def new_game(size):
    global game
    if 6 <= size <= 15:
        game = GameState(size)
        return jsonify(game.to_dict())
    return jsonify({'error': 'Invalid size'}), 400

@app.route('/api/state')
def get_state():
    if game is None:
        return jsonify({'error': 'No game started'}), 400
    return jsonify(game.to_dict())

@app.route('/api/toggle/<int:row>/<int:col>')
def toggle_mark(row, col):
    if game is None:
        return jsonify({'error': 'No game started'}), 400
    new_mark = game.toggle_mark(row, col)
    return jsonify({'mark': int(new_mark)})

@app.route('/')
def serve_landing():
    return send_from_directory('static', 'index.html')

@app.route('/game')
def serve_game():
    return send_from_directory('static', 'game.html')

if __name__ == '__main__':
    app.run(debug=True)