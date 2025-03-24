from flask import Flask, send_from_directory, jsonify, request, redirect, session
from flask_cors import CORS
import numpy as np
from board_generator import find_unique_solution_board
from typing import List, Set, Tuple, Optional
import pickle
import os
from datetime import timedelta
import secrets

app = Flask(__name__, static_url_path='/static')
CORS(app, supports_credentials=True)  # Enable credentials for session support
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key
app.permanent_session_lifetime = timedelta(days=1)  # Set session lifetime


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
        self.marks = np.zeros((self.n, self.n), dtype=int)
        
    def initialize_from_pickle(self, board_data: Tuple[np.ndarray, List[Tuple[int, int]]]):
        self.regions, self.queens = board_data
        self.marks = np.zeros((self.n, self.n), dtype=int)
        
    def to_dict(self):
        return {
            'regions': self.regions.tolist() if self.regions is not None else None,
            'marks': self.marks.tolist() if self.marks is not None else None,
            'size': self.n,
        }
        
    def from_dict(self, data):
        self.n = data['size']
        self.regions = np.array(data['regions']) if data['regions'] is not None else None
        self.marks = np.array(data['marks']) if data['marks'] is not None else None

    def reset_marks(self):
        """Reset all marks while preserving the board layout"""
        if self.marks is not None:
            self.marks = np.zeros((self.n, self.n), dtype=int)


class GameStateManager:
    def __init__(self):
        self.games_dir = "pregenerated_games"
        self.available_games = self._load_available_games()

    def _load_available_games(self):
        """Load available pre-generated games from organized subfolder structure"""
        available = {}
        for size in range(12, 16):
            size_dir = os.path.join(self.games_dir, f"board_size_{size}")
            available[size] = []
            
            if not os.path.exists(size_dir):
                continue
                
            for filename in sorted(os.listdir(size_dir)):
                if filename.endswith('.pkl'):
                    full_path = os.path.join(size_dir, filename)
                    available[size].append(full_path)
                    
        return available

    def get_game_state(self):
        """Get the current game state from session"""
        if 'game_state' not in session:
            return None
            
        game_state = GameState()
        game_state.from_dict(session['game_state'])
        return game_state

    def save_game_state(self, game_state):
        """Save the current game state to session"""
        session['game_state'] = game_state.to_dict()

    def create_new_game(self, size: int) -> GameState:
        """Create a new game of specified size"""
        game = GameState(size)
        game.initialize_from_generator()
        self.save_game_state(game)
        return game

    def load_specific_game(self, size: int, game_number: int) -> Optional[GameState]:
        """Load a specific pre-generated game"""
        pickle_path = os.path.join(self.games_dir, f'board_size_{size}', f'{game_number + 1}.pkl')
        
        if not os.path.exists(pickle_path):
            return None
            
        try:
            with open(pickle_path, 'rb') as f:
                board_data = pickle.load(f)
                
            game = GameState(size)
            game.regions, game.queens = board_data['board'], board_data['queens']
            game.marks = np.zeros((size, size), dtype=int)
            self.save_game_state(game)
            return game
        except Exception as e:
            print(f"Error loading game: {str(e)}")
            return None

    def reset_current_game(self) -> Optional[GameState]:
        """Reset the current game's marks while preserving the board layout"""
        game = self.get_game_state()
        if game is None:
            return None
            
        game.reset_marks()
        self.save_game_state(game)
        return game
    
# Initialize the game state manager
game_manager = GameStateManager()

@app.route('/api/select_game/<int:size>/<int:game_number>')
def select_specific_game(size, game_number):
    if size < 12 or size > 15:
        return jsonify({'error': 'Invalid size'}), 400
        
    game = game_manager.load_specific_game(size, game_number)
    if game is None:
        return jsonify({'error': 'Game not found'}), 404
        
    return jsonify(game.to_dict())

@app.route('/api/new_game/<int:size>')
def new_game(size):
    if size < 6 or size > 15:
        return jsonify({'error': 'Invalid size'}), 400
        
    if size >= 12:
        return redirect(f'/select_game/{size}')
        
    game = game_manager.create_new_game(size)
    return jsonify(game.to_dict())

@app.route('/api/state')
def get_state():
    game = game_manager.get_game_state()
    if game is None:
        return jsonify({'error': 'No game started'}), 400
    return jsonify(game.to_dict())

@app.route('/')
def serve_landing():
    return send_from_directory('static', 'index.html')

@app.route('/game')
def serve_game():
    return send_from_directory('static', 'game.html')

@app.route('/select_game/<int:size>')
def serve_game_selection(size):
    if size < 12 or size > 15:
        return redirect('/')
    return send_from_directory('static', 'select_game.html')

@app.route('/api/available_games/<int:size>')
def get_available_games(size):
    if size < 12 or size > 15:
        return jsonify({'error': 'Invalid size'}), 400
        
    size_dir = os.path.join('pregenerated_games', f'board_size_{size}')
    if not os.path.exists(size_dir):
        return jsonify({'size': size, 'games': []})
        
    game_numbers = []
    for f in os.listdir(size_dir):
        if f.endswith('.pkl'):
            try:
                game_num = int(f.split('.')[0])
                game_numbers.append(game_num)
            except ValueError:
                continue
    
    game_numbers.sort()
    return jsonify({
        'size': size,
        'games': game_numbers
    })

@app.route('/api/reset', methods=['POST'])
def reset_game():
    game = game_manager.reset_current_game()
    if game is None:
        return jsonify({'error': 'No game started'}), 400
        
    return jsonify(game.to_dict())

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)