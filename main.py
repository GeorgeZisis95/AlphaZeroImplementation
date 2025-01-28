from tictactoe import TicTacToe
import numpy as np

game = TicTacToe()
state = game.get_initial_state()
player = 1
print(game.get_valid_moves_indices(state))
action = np.random.choice(game.get_valid_moves_indices(state))
state = game.get_next_state(state, action, player)
game.render(state)
player = player * -1

while True:
    reward = game.get_reward(state, 1)
    if reward is not None:
        print(f"Game ends with reward: {reward}!")
        break
    best_score, action = game.minimax(state, 0, player==1)
    state = game.get_next_state(state, action, player)
    print(game.get_valid_moves_indices(state))
    game.render(state)
    player = player * -1