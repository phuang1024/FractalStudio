# Mandelbrot

Realtime visualization of the mandelbrot set.

Usage (you need a CUDA GPU):

```bash
cd ./src
make

# generate image
python main.py image --width 4096 --height 4096

# use pygame interactively
python main.py live --width 1280 --height 720
```

![](https://github.com/phuang1024/mandelbrot/blob/main/out.png?raw=true)
