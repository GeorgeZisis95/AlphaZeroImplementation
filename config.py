import argparse

parser = argparse.ArgumentParser(description="Pytorch Board Games Training")

# Monte Carlo Tree Search
parser.add_argument("--num_searches", default=100, type=int, help="Time given for a mcts search")
parser.add_argument("--cpuct", default=1.41, type=float, help="Exploration constant for ucb score")
parser.add_argument("--dirichlet_epsilon", default=0.25, type=float, help="Dirichlet equation, epsilon consant, lower means less noise")
parser.add_argument("--dirichlet_alpha", default=0.3, type=float, help="Dirichlet equation, alpha or eta constant")

args = parser.parse_args()