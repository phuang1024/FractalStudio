#!/bin/bash

XMIN=-2
XMAX=2
YMIN=-1
YMAX=1

g++ mandelbrot.cpp -o mandelbrot.out -O3 -Wall

./mandelbrot.out $XMIN $XMAX $YMIN $YMAX

python compile.py --bounds $XMIN:$XMAX:$YMIN:$YMAX
