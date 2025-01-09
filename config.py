import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Smallset Model")
    parser.add_argument('--mul_pos', action='store_true', help='Multiple positive')
    parser.add_argument('--no-mul_pos', action='store_false', dest='mul_pos', help='Single positives')
    parser.add_argument('--neg_remaining', action='store_true', help='Negative remaining')
    parser.add_argument('--time', type=str, required=True, help='Experiment time')
    parser.add_argument('--pos_ind', type=int, help='Positive step')
    parser.add_argument('--norm_type', type=str, required=True, help='Normalization type')
    parser.add_argument('--ff', action='store_true', help='Full features')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    return parser.parse_args()
