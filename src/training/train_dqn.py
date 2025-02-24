import numpy as np
from tqdm import tqdm
from ..core.game import Game
from ..models.dqn_model import DQNAgent
from ..utils.config import Config

def train_dqn(episodes=1000, target_update=10):
    """Train DQN agent."""
    agent = DQNAgent()
    
    rewards_history = []
    for episode in tqdm(range(episodes), desc="Training DQN"):
        game = Game()
        state = game.get_state()
        total_reward = 0
        
        while not game.is_game_over():
            # Get action from agent
            action = agent.get_move(state)
            if action is None:
                break
                
            # Make move and get reward
            old_state = state.copy()
            game.make_move(*action)
            next_state = game.get_state()
            
            # Calculate reward
            reward = 0
            if game.is_game_over():
                winner = game.get_winner()
                if winner == 1:  # Agent won
                    reward = 1.0
                elif winner == -1:  # Agent lost
                    reward = -1.0
                else:  # Draw
                    reward = 0.1
            
            # Store experience
            agent.remember(old_state, action, reward, next_state, game.is_game_over())
            
            # Train agent
            loss = agent.train()
            if loss:
                total_reward += reward
            
            state = next_state
            
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
            
        rewards_history.append(total_reward)
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"{Config.CHECKPOINT_DIR}/dqn_model_episode_{episode+1}.pt")
            print(f"Average reward over last 100 episodes: {np.mean(rewards_history[-100:]):.2f}")
    
    # Save final model
    agent.save(f"{Config.CHECKPOINT_DIR}/dqn_final.pt")
    return rewards_history

def evaluate_dqn(agent, num_games=100):
    """Evaluate trained DQN agent."""
    wins = 0
    draws = 0
    
    for _ in tqdm(range(num_games), desc="Evaluating DQN"):
        game = Game()
        state = game.get_state()
        
        while not game.is_game_over():
            action = agent.get_move(state)
            if action is None:
                break
            game.make_move(*action)
            state = game.get_state()
        
        winner = game.get_winner()
        if winner == 1:  # Agent won
            wins += 1
        elif winner == 0:  # Draw
            draws += 1
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    
    print(f"\nEvaluation Results:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Draw Rate: {draw_rate:.2%}")
    print(f"Loss Rate: {1 - win_rate - draw_rate:.2%}")
    
    return win_rate, draw_rate