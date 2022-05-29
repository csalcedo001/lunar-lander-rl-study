from argparse import ArgumentParser

def get_parser(episodes=100, max_iter=1000)
    parser = ArgumentParser()

    parser.add_argument(
        '--episodes',
        type=int, default=1000)
    parser.add_argument(
        '--max-iter',
        type=int, default=1000)
    parser.add_argument(
        '--no-render',
        default=False, action="store_const", const=True)

    return parser
