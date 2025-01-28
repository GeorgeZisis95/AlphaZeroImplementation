import warnings
# There's an annoying warning when using CuDNN and this removes it
warnings.filterwarnings("ignore", ".*Applied workaround*.",)

import numpy as np
import torch

from tictactoe import TicTacToe
from model import ResNet

# Initialize game
game = TicTacToe()
state = game.get_initial_state()
player = 1

# Get random first action
action = np.random.choice(game.get_valid_moves_indices(state))
state = game.get_next_state(state, action, player)
game.render(state)
player = player * -1

# Check model
encoded_state = game.get_encoded_state(state)
print(encoded_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device we use is: {device}")

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
print(tensor_state)

model = ResNet(game, device=device, input_channels=3, filters=64, res_blocks=9)

policy, value = model(tensor_state)

value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(f"The value of current state is: {value}")
print(f"The policy of the current state is: {policy}")