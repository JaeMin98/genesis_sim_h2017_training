import torch

# 하이퍼 파라미터 설정
ALGORITHM = "SAC"
TOTAL_TIMESTEPS = 30_000_000
SEED = 42
NUM_ENVS = 1
LEARNING_RATE = 0.0003
BATCH_SIZE = 1024
GAMMA = 0.977
BUFFER_SIZE = 30_000_000
LEARNING_STARTS = 4096
TRAIN_FREQ = 10
GRADIENT_STEPS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
ENT_COEF = "auto"
SAVE_FREQ = 1000

#커리큘럼 러닝
REPLAY_RATIO = 0.5

#Environment
GOAL_ALLOWABLE_ERROR = 0.03
ACTION_WEIGHT = 0.3
RD_WEIGHT = -1.0  # distance_reward
RP_WEIGHT = 2.0  # progress_reward
RE_WEIGHT = -0.05  # efficiency_reward
RS_WEIGHT = 50  # success_reward