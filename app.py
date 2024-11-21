from flask import Flask, send_from_directory, jsonify, request, redirect
from flask_cors import CORS
import numpy as np
from board_generator import find_unique_solution_board
from typing import List, Set, Tuple, Optional

import pickle
import os

app = Flask(__name__, static_url_path='/static')
CORS(app)


class GameState:
    def __init__(self, n=8):
        self.n = n
        self.queens = None
        self.regions = None
        self.marks = None
        
    def initialize_from_generator(self):
        self.regions, self.queens = find_unique_solution_board(self.n)
        print("regions:", self.regions)
        print("queens:", self.queens)
        self.marks = np.zeros((self.n, self.n), dtype=int) # 0: unmarked, 1: X, 2: queen
        
    def initialize_from_pickle(self, board_data: Tuple[np.ndarray, List[Tuple[int, int]]]):
        self.regions, self.queens = board_data
        self.marks = np.zeros((self.n, self.n), dtype=int)
        
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
available_games = {}  # size -> list of game files

def load_available_games():
    """Load available pre-generated games from organized subfolder structure"""
    games_dir = "pregenerated_games"
    
    for size in range(12, 16):
        size_dir = os.path.join(games_dir, f"board_size_{size}")
        available_games[size] = []
        
        # Skip if directory doesn't exist
        if not os.path.exists(size_dir):
            continue
            
        # Look for numbered pickle files in the size directory
        for filename in sorted(os.listdir(size_dir)):
            if filename.endswith('.pkl'):
                full_path = os.path.join(size_dir, filename)
                available_games[size].append(full_path)
                print(f"Loaded game file: {full_path}")
                
        print(f"Found {len(available_games[size])} games for size {size}")


@app.route('/api/select_game/<int:size>/<int:game_number>')
def select_specific_game(size, game_number):
    """Handle selection of specific pre-generated game"""
    if size < 12 or size > 15:
        return jsonify({'error': 'Invalid size'}), 400
    
    # Construct path to the specific pickle file
    pickle_path = os.path.join('pregenerated_games', f'board_size_{size}', f'{game_number + 1}.pkl')
    
    print("Board pickle path: ", pickle_path)

    if not os.path.exists(pickle_path):
        return jsonify({'error': 'Game not found'}), 404
    
    global game
    game = GameState(size)
    
    try:
        with open(pickle_path, 'rb') as f:
            board_data = pickle.load(f)
        
        print(f"Board Data: {board_data}")

        game.regions, game.queens = board_data['board'], board_data['queens']
        game.marks = np.zeros((size, size), dtype=int)
        return jsonify(game.to_dict())
    except Exception as e:
        print(f"Error loading game: {str(e)}")  # For debugging
        return jsonify({'error': f'Failed to load game: {str(e)}'}), 500
    

@app.route('/api/new_game/<int:size>')
def new_game(size):
    """Modified to handle redirection for larger board sizes"""
    if size < 6 or size > 15:
        return jsonify({'error': 'Invalid size'}), 400
        
    # For sizes 12-15, redirect to game selection
    if size >= 12:
        return redirect(f'/select_game/{size}')
        
    # For smaller sizes, generate new game as before
    global game
    game = GameState(size)
    game.initialize_from_generator()
    return jsonify(game.to_dict())


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

@app.route('/select_game/<int:size>')
def serve_game_selection(size):
    """Serve the game selection page for sizes 12-15"""
    # Validate size range
    if size < 12 or size > 15:
        return redirect('/')
    return send_from_directory('static', 'select_game.html')


@app.route('/api/available_games/<int:size>')
def get_available_games(size):
    """API endpoint to get available game numbers for given size"""
    if size < 12 or size > 15:
        return jsonify({'error': 'Invalid size'}), 400
        
    # Get the size's directory
    size_dir = os.path.join('pregenerated_games', f'board_size_{size}')
    if not os.path.exists(size_dir):
        return jsonify({'size': size, 'games': []})
        
    # Get all .pkl files and extract their numbers
    game_numbers = []
    for f in os.listdir(size_dir):
        if f.endswith('.pkl'):
            # Extract number from filename (e.g., "1.pkl" -> 1)
            try:
                game_num = int(f.split('.')[0])
                game_numbers.append(game_num)
            except ValueError:
                continue
    
    # Sort numbers for consistent display
    game_numbers.sort()
    
    return jsonify({
        'size': size,
        'games': game_numbers
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)