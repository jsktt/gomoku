import numpy as np
import torch
from tqdm import tqdm
from src.core.game import Game
from src.models.cnn_model import GomokuCNN
from src.models.mcts_model import MCTS
from src.models.dqn_model import DQNAgent

class RandomPlayer:
    def get_move(self, board_state):
        valid_moves = [(i, j) for i in range(15) for j in range(15) 
                      if board_state[i, j] == 0]
        return valid_moves[np.random.randint(len(valid_moves))] if valid_moves else None

def play_game(player1, player2, render=False):
    """Play a single game between two players."""
    game = Game()
    current_player = player1
    
    while not game.is_game_over():
        state = game.get_state()
        if render:
            print(game.board)
            
        move = current_player.get_move(state)
        if move is None:
            break
            
        game.make_move(*move)
        current_player = player2 if current_player == player1 else player1
    
    winner = game.get_winner()
    return winner

def evaluate_winrate(player1, player2, num_games=100):
    """Evaluate win rate between two players."""
    player1_wins = 0
    player2_wins = 0
    draws = 0
    
    for i in tqdm(range(num_games), desc="Playing games"):
        # Alternate who goes first
        if i % 2 == 0:
            winner = play_game(player1, player2)
            if winner == 1:
                player1_wins += 1
            elif winner == -1:
                player2_wins += 1
            else:
                draws += 1
        else:
            winner = play_game(player2, player1)
            if winner == 1:
                player2_wins += 1
            elif winner == -1:
                player1_wins += 1
            else:
                draws += 1
    
    return {
        'player1_winrate': player1_wins / num_games,
        'player2_winrate': player2_wins / num_games,
        'draw_rate': draws / num_games
    }

def main():
    # Initialize models
    print("Initializing models...")
    cnn_model = GomokuCNN()
    mcts_player = MCTS(n_simulations=500)
    dqn_agent = DQNAgent()
    random_player = RandomPlayer()
    
    # Load trained models if available
    try:
        cnn_model.load_state_dict(torch.load('data/models/cnn_final.pt'))
        dqn_agent.load('data/models/dqn_final.pt')
    except FileNotFoundError:
        print("Warning: Could not find model files. Using untrained models.")
    
    # Test configurations
    matchups = [
        (cnn_model, random_player, "CNN vs Random"),
        (dqn_agent, random_player, "DQN vs Random"),
        (cnn_model, mcts_player, "CNN vs MCTS"),
        (cnn_model, dqn_agent, "CNN vs DQN"),
        (mcts_player, dqn_agent, "MCTS vs DQN")
    ]
    
    # Run evaluations
    print("\nEvaluating win rates...")
    results = {}
    for player1, player2, name in matchups:
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_winrate(player1, player2, num_games=50)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    for matchup, stats in results.items():
        print(f"\n{matchup}:")
        print(f"Player 1 Win Rate: {stats['player1_winrate']:.2%}")
        print(f"Player 2 Win Rate: {stats['player2_winrate']:.2%}")
        print(f"Draw Rate: {stats['draw_rate']:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    main()