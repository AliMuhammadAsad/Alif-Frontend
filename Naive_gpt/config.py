# Model/config.py
import torch

# Define hyperparameters and constants
BATCH_SIZE = 16
BLOCK_SIZE = 1024
MAX_ITERS = 1
EVAL_INTERVAL = 500
LEARNING_RATE = 6e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
DROPOUT = 0.2
MODEL_PATH = "Naive_gpt\model_weights_llama"  # Where to save weights
