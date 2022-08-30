import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epoch", type=int, default=100, help="the number of epoch to train the agent")
    parser.add_argument("--n-episode", type=int, default=200, help="the number of episode to train the agent")
    parser.add_argument("--seed", type=int, default=123, help="the random seed")
    parser.add_argument("--replay-strategy", type=str, default="future", help="the HER strategy")
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument("--buffer-size", type=int, default=1e6, help="the buffer size to store sample")
    parser.add_argument("--batch-size", type=int, default=256, help="the sample size to update the agent")
    parser.add_argument("--gamma", type=float, default=0.98, help="the discount factor")
    parser.add_argument("--lr-actor", type=float, default=0.001, help="learning rate for actor")
    parser.add_argument("--lr-critic", type=float, default=0.001, help="learning rate for critic")
    parser.add_argument("--tau", type=float, default=0.05, help="the soft update parameter")
    parser.add_argument("--save-dir", type=str, default="saved_models/", help="the path to save your model")
    parser.add_argument("--max-time-step", type=int, default=50, help="maximum simulation timestep per episode")
    parser.add_argument("--hidden-dim", type=int, default=256, help="the hidden dim for the deep neural network")
    parser.add_argument("--device", type=str, default="cuda:0", help="whether use gpu")

    args = parser.parse_args()
    return args
