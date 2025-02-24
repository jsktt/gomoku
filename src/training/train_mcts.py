import numpy as np
from tqdm import tqdm
from ..models.mcts_model import MCTS, MCTSNode
from ..core.game import Game

def self_play_game(mcts_player: MCTS) -> tuple:
    """Play a game using MCTS against itself."""
    game = Game()
    states, policies, values = [], [], []
    max_moves = 225  # Maximum possible moves on 15x15 board
    moves_count = 0
    
    while not game.is_game_over() and moves_count < max_moves:
        moves_count += 1
        state = game.get_state()
        states.append(state)
        
        root = MCTSNode(state)
        root.expand(game)
        
        for _ in range(mcts_player.n_simulations):
            game_copy = Game()
            game_copy.board.board = state.copy()
            mcts_player._playout(root, game_copy)
        
        policy = np.zeros(225)  # 15x15 board
        total_visits = sum(child.n_visits for child in root.children.values())
        
        if total_visits > 0:
            for move, child in root.children.items():
                policy[move[0] * 15 + move[1]] = child.n_visits / total_visits
        
        policies.append(policy)
        
        if root.children:
            move = max(root.children.items(), key=lambda x: x[1].n_visits)[0]
            game.make_move(*move)
        else:
            break
    
    winner = game.get_winner()
    values = [winner * (1 if i % 2 == 0 else -1) for i in range(len(states))]
    
    return states, policies, values

def train_mcts(num_games=10, n_simulations=100):
    """Generate training data using MCTS self-play."""
    mcts_player = MCTS(n_simulations=n_simulations)
    all_states, all_policies, all_values = [], [], []
    
    for _ in tqdm(range(num_games), desc="Self-play games"):
        states, policies, values = self_play_game(mcts_player)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
    
    return (np.array(all_states, dtype=np.float32),
            np.array(all_policies, dtype=np.float32),
            np.array(all_values, dtype=np.float32))