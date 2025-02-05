import warnings
# There's an annoying warning when using CuDNN and this removes it
warnings.filterwarnings("ignore", ".*Applied workaround*.",)

import numpy as np
import torch
import multiprocessing

from tictactoe import TicTacToe
from model import ResNet
from mcts import MonteCarloTreeSearch
from agent import AlphaZero
from config import args

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)    
    np.set_printoptions(suppress=True)

    # Initialize game
    game = TicTacToe()

    print("===> Building Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, device, input_channels=args.input_channels, filters=args.filters, res_blocks=args.res_blocks)
    print(f"...Device: {device}")
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("===> Getting the correct iteration...")
    if args.iteration != 1:
        model.load_state_dict(torch.load(f"./model/{repr(game)}{args.version}.{args.iteration - 1}.pt"))
        optimizer.load_state_dict(torch.load(f"./optim/{repr(game)}{args.version}.{args.iteration - 1}.pt"))
        print(f"Loaded model {repr(game)}{args.version}.{args.iteration - 1}pt")

    mcts = MonteCarloTreeSearch(game, args, model)
    agent = AlphaZero(model, optimizer, game, args, mcts)

    if args.get_dataset:
        agent.create_dataset(parallel=args.parallelize)

    if args.train_model:
        dataset = agent.include_history()
        if args.scheduler:
            agent.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr, eta_min=0.001)
        agent.train_dataset(dataset=dataset)