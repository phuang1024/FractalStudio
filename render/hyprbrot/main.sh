#!/bin/bash

XMIN=-2.61
XMAX=-0.11
YMIN=0
YMAX=1.25

echo Compiling programs.
g++ mandelbrot.cpp -o mandelbrot.out -O3 -Wall
g++ buddhabrot.cpp -o buddhabrot.out -O3 -Wall

echo Computing mandelbrot.
./mandelbrot.out $XMIN $XMAX $YMIN $YMAX
echo Computing buddhabrot.
./buddhabrot.out $XMIN $XMAX $YMIN $YMAX

echo Making final image.
python compile.py
