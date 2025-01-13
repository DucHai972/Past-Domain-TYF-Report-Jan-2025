import argparse
from dataclasses import dataclass

@dataclass
class Config:
    time: str
    pos_ind: int
    norm_type: str
    lr: float
    csv_path: str
    small_set: bool
    model: str
    alpha: float = 0.85
    epoch: int = 100
    

def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="Train and Evaluate Model")
    parser.add_argument('--time', type=str, required=True, help='Time value (e.g., 1_FE)')
    parser.add_argument('--pos_ind', type=int, required=False, default=1, help='Positive step')
    parser.add_argument('--norm_type', type=str, required=True, help='Type of normalization (new/old)')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    parser.add_argument('--small_set', action='store_true', help='Use small set for testing')   
    parser.add_argument('--model', type=str, default='resnet', help='Model type')
    # python main.py --time test_t2_newnorm --norm_type new --lr 1e-7 --pos_ind 2 > ./out/test_t4_newnorm.out 2>&1 && \
    # python main.py --time test_t8_newnorm --norm_type new --lr 1e-7 --pos_ind 8 > ./out/test_t8_newnorm.out 2>&1 
    
    # python main.py --time cnn_t6_newnorm_lr1e7 --norm_type new --lr 1e-7 --pos_ind 6 --model cnn2d > ./out/cnn_t6_newnorm_lr1e7.out 2>&1 && \
    # python main.py --time cnn_t8_newnorm_lr1e7 --norm_type new --lr 1e-7 --pos_ind 8 --model cnn2d > ./out/cnn_t8_newnorm_lr1e7.out 2>&1

    args = parser.parse_args()


    args.csv_path = f"/N/slate/tnn3/HaiND/01-06_report/csv/merra_full.csv"

    config = Config(
        time=args.time,
        pos_ind=args.pos_ind,
        norm_type=args.norm_type,
        lr=args.lr,
        csv_path=args.csv_path,
        small_set=args.small_set,
        model=args.model
    )

    if config.small_set:
        config.epoch = 5

    print(f'time: {config.time}')
    print(f'pos_ind: {config.pos_ind}')
    print(f'norm_type: {config.norm_type}')
    print(f'lr: {config.lr}')
    print(f'csv_path: {config.csv_path}')
    print(f'alpha: {config.alpha}')
    print(f'epoch: {config.epoch}')
    print(f'small_set: {config.small_set}')
    print(f'model: {config.model}')

    return config