import warnings
# There's an annoying warning when using CuDNN and this removes it
warnings.filterwarnings("ignore", ".*Applied workaround*.",)

import numpy as np
import torch

from tictactoe import TicTacToe
from model import ResNet
from mcts import MonteCarloTreeSearch

from config import args

print(f"Args: {args}")

# Initialize game
game = TicTacToe()
state = game.get_initial_state()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device we use is: {device}")

model = ResNet(game, device=device, input_channels=3, filters=64, res_blocks=9)
mcts = MonteCarloTreeSearch(game=game, args=args, model=model)

state = game.get_initial_state()
player = 1
while True:
    reward = game.get_reward(state, player=1)
    if reward is not None:
        print(f"Game ends with reward: {reward}!")
        print(f"In state:")
        game.render(state)
        break
    if player == 1:
        root = mcts.search(state, player)
        action_probs = root.get_action_probs(game, temperature=0.0)
    else:
        root = mcts.search(state, player)
        action_probs = root.get_action_probs(game, temperature=1.0)
    print(f"Action Probabilities:{action_probs}")
    print(f"For state:")
    game.render(state)
    
    action = np.random.choice(game.action_size, p=action_probs)
    state = game.get_next_state(state, action, player)
    
    player = player * -1

# Just with mcts and the resnet the agents seem to defend correctly with probs=1 when
# they are about to lose