class Config:
    # Game settings
    BOARD_SIZE = 15
    
    # CNN settings
    CNN_CHANNELS = [64, 128, 256]
    CNN_KERNEL_SIZE = 3
    CNN_LEARNING_RATE = 0.001
    CNN_BATCH_SIZE = 32
    
    # MCTS settings
    MCTS_SIMULATIONS = 800
    MCTS_C_PUCT = 1.0
    MCTS_TEMPERATURE = 1.0
    MCTS_TEMPERATURE_THRESHOLD = 30
    
    # DQN settings
    DQN_LEARNING_RATE = 0.001
    DQN_BATCH_SIZE = 32
    DQN_MEMORY_SIZE = 10000
    DQN_TARGET_UPDATE = 1000
    DQN_GAMMA = 0.99
    
    # Training settings
    NUM_EPOCHS = 100
    SAVE_INTERVAL = 10
    CHECKPOINT_DIR = "data/models"