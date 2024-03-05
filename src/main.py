import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices={"viewer", "render"})
    args = parser.parse_args()


if __name__ == "__main__":
    main()
