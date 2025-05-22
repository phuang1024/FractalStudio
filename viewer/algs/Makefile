NVCC = nvcc
CFLAGS = -use_fast_math -O3 -Xptxas -O3

cudabrot:
	$(NVCC) buddha.cu $(CFLAGS) -o buddha.out
	$(NVCC) mandel.cu $(CFLAGS) -o mandel.out
