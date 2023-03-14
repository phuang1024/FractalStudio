# Mandelbrot

Realtime visualization of the mandelbrot set.

## General usage:

```bash
cd /path/to/mandelbrot/src

# If you have a CUDA gpu
make cuda
# or use cpu
make cpu

# generate image
python main.py image --width 4096 --height 4096

# use pygame interactively
python main.py live --width 1280 --height 720

# change kernels
python main.py --kernel cpu
```

## Options

- `mode`: `image` or `live`.
- `--kernel`: `cpu` or `cuda` to set computing kernel. GPU is usually faster.
- `--width`: Width of viewing window or generated image.
- `--height`: Height
- `--max-iters`: Max iters to simulate function. More iters is slower but more accurate.

![](https://github.com/phuang1024/mandelbrot/blob/main/out.png?raw=true)
