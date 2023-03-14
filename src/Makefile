.PHONY: cuda cpu

# Usage:
# ./a.out width height max_iters
# Then send "x_start x_end y_start y_end\n" to stdin
# Read result from stdout.

cuda:
	nvcc kernel.cu -o kernel.cuda.out

cpu:
	g++ kernel.cpp -Wall -O3 -pthread -o kernel.cpu.out
