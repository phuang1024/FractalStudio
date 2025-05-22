import argparse

from algs.buddhabrot import Buddhabrot
from algs.buddhacu import Buddhacu
from algs.mandelbrot import Mandelbrot
from algs.mandelcu import Mandelcu
from algs.nebulabrot import Nebulabrot
from algs.nebulacu import Nebulacu
from algs.test import *
from viewer import viewer


algorithms = {
    "mandel": Mandelbrot,
    "mandelcu": Mandelcu,
    "buddha": Buddhabrot,
    "nebula": Nebulabrot,
    "buddhacu": Buddhacu,
    "nebulacu": Nebulacu,
    "test_solid": SolidColor,
    "test_image": ImageResize,
}


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("action", choices={"viewer", "render"})
    parser.add_argument("--alg", choices=algorithms.keys(), required=True)
    # TODO this is unwieldy for strings.
    parser.add_argument("--init-args", nargs="*", help="Format: arg='str'  arg=2")
    args = parser.parse_args()

    init_args = {}
    if args.init_args:
        for arg in args.init_args:
            k, v = arg.split("=")
            init_args[k] = eval(v)
    alg = algorithms[args.alg](**init_args)

    #if args.action == "viewer":
    viewer(args, alg)


if __name__ == "__main__":
    main()
