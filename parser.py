from argparse import ArgumentParser

def get_parser(episodes=100, max_iter=1000):
    parser = ArgumentParser()

    parser.add_argument(
        'agent',
        type=str)
    parser.add_argument(
        '--env',
        type=str, default='LunarLander-v2')
    parser.add_argument(
        '--checkpoint',
        type=str, default=None)
    parser.add_argument(
        '--episodes',
        type=int, default=episodes)
    parser.add_argument(
        '--max-iter',
        type=int, default=max_iter)
    parser.add_argument(
        '--no-render',
        default=False, action='store_const', const=True)

    return parser
