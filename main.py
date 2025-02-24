# main.py

import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Import models
from src.models.cnn_model import GomokuCNN
from src.models.mcts_model import MCTS
from src.models.dqn_model import DQNAgent

# Import training functions
from src.training.train_cnn import train_cnn
from src.training.train_mcts import train_mcts
from src.training.train_dqn import train_dqn, evaluate_dqn

# Import utilities
from src.core.game import Game
from src.utils.config import Config

def setup_directories():
    """Create necessary directories."""
    directories = [
        'data/models',
        'data/raw',
        'data/processed',
        'logs'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def generate_training_data(num_games=1000):
    """Generate initial training data through self-play."""
    states, policies, values = [], [], []
    game = Game()
    
    for _ in tqdm(range(num_games), desc="Generating training data"):
        game_states = []
        game_policies = []
        current_player = 1
        
        while not game.is_game_over():
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            
            # Create simple policy (uniform distribution over valid moves)
            policy = np.zeros(Config.BOARD_SIZE * Config.BOARD_SIZE, dtype=np.float32)
            for move in valid_moves:
                policy[move[0] * Config.BOARD_SIZE + move[1]] = 1.0
            policy = policy / len(valid_moves) if valid_moves else policy
            
            # Store state and policy
            game_states.append(state)
            game_policies.append(policy)
            
            # Make random move
            if valid_moves:
                x, y = valid_moves[np.random.randint(len(valid_moves))]
                game.make_move(x, y)
            else:
                break
        
        # Get game outcome
        winner = game.get_winner()
        game_values = [float(winner * current_player) for _ in game_states]
        
        # Store game data
        states.extend(game_states)
        policies.extend(game_policies)
        values.extend(game_values)
        
        # Reset game for next iteration
        game = Game()
    
    states = np.array(states, dtype=np.float32)
    policies = np.array(policies, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    
    # Print shapes for debugging
    print(f"Generated data shapes:")
    print(f"States: {states.shape}")
    print(f"Policies: {policies.shape}")
    print(f"Values: {values.shape}")
    
    return states, policies, values

def train_models(args):
    """Train selected models."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.model == 'cnn' or args.model == 'all':
        print("\nTraining CNN model...")
        cnn_model = GomokuCNN()
        train_data = generate_training_data(num_games=args.num_games)
        train_cnn(cnn_model, train_data)
        torch.save(cnn_model.state_dict(), 'data/models/cnn_final.pt')
    '''
     if args.model == 'mcts' or args.model == 'all':
        print("\nTraining MCTS model...")
        mcts_player = MCTS(n_simulations=args.mcts_sims)
        states, policies, values = train_mcts(num_games=args.num_games)
        # MCTS doesn't need to save model as it's not a learning model
    
    '''
    if args.model == 'dqn' or args.model == 'all':
        print("\nTraining DQN model...")
        rewards = train_dqn(episodes=args.num_games)
        print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")

def evaluate_models(args):
    """Evaluate trained models."""
    def play_game(model, num_games=200):
        wins = 0
        for _ in tqdm(range(num_games), desc="Evaluating"):
            game = Game()
            while not game.is_game_over():
                state = game.get_state()
                move = model.get_move(state)
                if move is None:
                    break
                game.make_move(*move)
            if game.get_winner() == 1:
                wins += 1
        return wins / num_games
    
    results = {}
    
    if args.model == 'cnn' or args.model == 'all':
        cnn_model = GomokuCNN()
        cnn_model.load_state_dict(torch.load('data/models/cnn_final.pt'))
        results['CNN'] = play_game(cnn_model, args.eval_games)
    '''
    if args.model == 'mcts' or args.model == 'all':
        mcts_player = MCTS(n_simulations=args.mcts_sims)
        results['MCTS'] = play_game(mcts_player, args.eval_games)
    '''
    if args.model == 'dqn' or args.model == 'all':
        dqn_agent = DQNAgent()
        dqn_agent.load('data/models/dqn_final.pt')
        results['DQN'] = play_game(dqn_agent, args.eval_games)
    
    print("\nEvaluation Results:")
    for model, win_rate in results.items():
        print(f"{model} Win Rate: {win_rate:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Gomoku AI Training and Evaluation')
    parser.add_argument('--model', type=str, default='all', choices=['cnn', 'mcts', 'dqn', 'all'],
                        help='Model to train/evaluate')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                        help='Mode of operation')
    parser.add_argument('--num_games', type=int, default=1000,
                        help='Number of games for training')
    parser.add_argument('--eval_games', type=int, default=100,
                        help='Number of games for evaluation')
    parser.add_argument('--mcts_sims', type=int, default=800,
                        help='Number of MCTS simulations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Train and/or evaluate models
    if args.mode in ['train', 'both']:
        train_models(args)
    if args.mode in ['eval', 'both']:
        evaluate_models(args)

if __name__ == "__main__":
    main()