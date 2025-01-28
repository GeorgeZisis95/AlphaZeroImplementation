import argparse

parser = argparse.ArgumentParser(description="Pytorch Board Games Training")

# Monte Carlo Tree Search
parser.add_argument("--num_searches", default=100, type=int, help="Time given for a mcts search")
parser.add_argument("--cpuct", default=1.41, type=float, help="Exploration constant for ucb score")
parser.add_argument("--dirichlet_epsilon", default=0.25, type=float, help="Dirichlet equation, epsilon consant, lower means less noise")
parser.add_argument("--dirichlet_alpha", default=1.25, type=float, help="Dirichlet equation, alpha or eta constant")

# Alpha Zero Agent - Selfplay
parser.add_argument("--temperature", default=1.0, type=float, help="Starting temperature constant")
# Alpha Zero Agent - Create dataset
parser.add_argument("--sp_iters", type=int, default=50, help="Number of self play games per iteration")
parser.add_argument("--workers", default=8, type=int, help="Number of workers to work in parallel")
parser.add_argument("--version", default=1, type=int,  help="Dataset Version")
parser.add_argument("--iteration", default=1, type=int,  help="Current iteration")
# Alpha Zero Agent - Train
parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
# Alpha Zero Agent - Train Dataset
parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs for trainining")
# Alpha Zero Agent - Include History
parser.add_argument("--history", default=0, type=int,  help="Number of past datasets to include")
parser.add_argument("--dataset_decay_rate", default=2.0, type=float,  help="Decay ratio of elements of past datasets to combine with new dataset")

# Main - Model
parser.add_argument("--input_channels", default=3, type=int, help="Input channels for Residual Model")
parser.add_argument("--filters", default=64, type=int, help="Number of filters for Residual Model")
parser.add_argument("--res_blocks", default=4, type=int, help="Number of blocks for ResTower")
# Main - Optimizer
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay for optimizer")

# Main - Get Dataset
parser.add_argument("--parallelize", "-p", action="store_true", help="parallelize selfplay games")
parser.add_argument("--get_dataset", "-d", action="store_true", help="start selfplay process")
# Main - Train Model
parser.add_argument("--train_model", "-t", action="store_true", help="start training the model")
parser.add_argument("--scheduler", "-r", action="store_true", help="load scheduler")

args = parser.parse_args()