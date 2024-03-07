# FractalStudio

![](https://github.com/phuang1024/FractalStudio/blob/master/gallery/ViewerBuddhabrot.jpg?raw=true)

Cuda accelerated fractal rendering.

[Gallery](https://github.com/phuang1024/FractalStudio/tree/master/gallery)


## Viewer

The viewer is an interactive GUI fractal renderer.

The viewer uses PyTorch cuda vector code to render fractals efficiently with
only python.

```bash
cd ./viewer

python main.py viewer --alg mandelbrot
python main.py viewer --alg buddhabrot
python main.py viewer --alg nebulabrot
```


## Renderer

This uses native C and Cuda code to render fractals. It is designed to export
high quality images, but also includes a live viewer for the mandelbrot.

This is faster than the `viewer`, but is harder to use.

This is the first version of this project, which can still be found on the `old`
branch.

### `render/mandelbrot/`

Realtime visualization of the mandelbrot set.

```bash
cd ./render/mandelbrot

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

**Options** to `main.py`:

- `mode`: `image` or `live`.
- `--kernel`: `cpu` or `cuda` to set computing kernel. GPU is usually faster.
- `--width`: Width of viewing window or generated image.
- `--height`: Height
- `--max-iters`: Max iters to simulate function. More iters is slower but more accurate.

### `render/buddhabrot/`

Render the [buddhabrot](https://en.wikipedia.org/wiki/Buddhabrot) or nebulabrot.

```bash
cd ./render/buddhabrot

make cpu

# generate buddhabrot
# Args are [iters] [samples]
./a.out 1000 10000000
python convert.py out.img buddhabrot.png

# generate nebulabrot
# calls a.out, convert.py, nebula.py many times and compiles the final image.
./nebula.sh
```
