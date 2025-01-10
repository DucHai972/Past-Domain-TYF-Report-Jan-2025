import argparse
from dataclasses import dataclass

@dataclass
class Config:
    mul_pos: bool
    neg_remaining: bool
    time: str
    pos_ind: int
    norm_type: str
    ff: bool
    chanh: bool
    aug: bool
    lr: float
    csv_path: str
    alpha: float = 0.85
    epoch: int = 200

def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="Train and Evaluate Model")
    parser.add_argument('--mul_pos', action='store_true', help='Multiple positive (True/False)')
    parser.add_argument('--no-mul_pos', action='store_false', dest='mul_pos', help='Single positives')
    parser.add_argument('--neg_remaining', action='store_true', help='Neg remaining (True/False)')
    parser.add_argument('--no-neg_remaining', action='store_false', dest='neg_remaining', help='Neg from t-20 to t-40')
    parser.add_argument('--time', type=str, required=True, help='Time value (e.g., 1_FE)')
    parser.add_argument('--pos_ind', type=int, required=False, default=1, help='Positive step')
    parser.add_argument('--norm_type', type=str, required=True, help='Type of normalization (new/old)')
    parser.add_argument('--ff', action='store_true', help='Full features')
    parser.add_argument('--no-ff', action='store_false', dest='ff', help='No full features')
    parser.add_argument('--chanh', action='store_true', help='Features thay Chanh')
    parser.add_argument('--no-chanh', action='store_false', dest='chanh', help='No features thay Chanh')
    parser.add_argument('--aug', action='store_true', help='Augmentation')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')

    # python main.py --mul_pos --neg_remaining --time test_t2_newnorm --norm_type new --ff --chanh --lr 1e-7 

    args = parser.parse_args()

    if args.aug is None:
        args.aug = False

    if args.mul_pos:
        args.csv_path = '/N/slate/tnn3/HaiND/11-17_newPast/newpast_merra_full_mulpos.csv'
    else:
        args.csv_path = '/N/slate/tnn3/HaiND/11-17_newPast/newpast_merra_full.csv'

    config = Config(
        mul_pos=args.mul_pos,
        neg_remaining=args.neg_remaining,
        time=args.time,
        pos_ind=args.pos_ind,
        norm_type=args.norm_type,
        ff=args.ff,
        chanh=args.chanh,
        aug=args.aug,
        lr=args.lr,
        csv_path=args.csv_path
    )

    print(f'mul_pos: {config.mul_pos}')
    print(f'neg_remaining: {config.neg_remaining}')
    print(f'time: {config.time}')
    print(f'pos_ind: {config.pos_ind}')
    print(f'norm_type: {config.norm_type}')
    print(f'ff: {config.ff}')
    print(f'chanh: {config.chanh}')
    print(f'aug: {config.aug}')
    print(f'lr: {config.lr}')
    print(f'csv_path: {config.csv_path}')
    print(f'alpha: {config.alpha}')
    print(f'epoch: {config.epoch}')

    return config