.PHONY: cuda cpu

cuda:
	nvcc kernel.cu -Wall -O3 -o kernel.cuda.out

cpu:
	g++ kernel.cpp -Wall -O3 -o kernel.cpu.out
