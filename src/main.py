import argparse

from algs.test import TestFractal
from viewer import viewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices={"viewer", "render"})
    parser.add_argument("--alg", choices={"test"}, required=True)
    args = parser.parse_args()

    classes = {
        "test": TestFractal,
    }
    algorithm = classes[args.alg]()

    if args.action == "viewer":
        viewer(args, algorithm)


if __name__ == "__main__":
    main()
