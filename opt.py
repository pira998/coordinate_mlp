import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # Image path with help of argparse required=True
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')

    # network structure with choices identity, mlp, pe
    parser.add_argument('--arch', type=str, default='identity', choices=['identity','pe','ff', 'siren'], help='Network architecture')

    # gaussian embedding scale factor
    parser.add_argument('--sc', type=float, default=10., help='Gaussian embedding scale factor')

    # omega for siren
    parser.add_argument('--omega', type=float, default=30., help='Omega for siren')

    # batch size with 4
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')

    # epochs with 10
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')

    # learning rate with 0.001
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    # experiment name
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')

    return parser.parse_args()






