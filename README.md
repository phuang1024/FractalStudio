# FractalStudio

![](./gallery/ViewerCudabrot.gif?raw=true)

Cuda accelerated fractal rendering.

[Gallery](./gallery/)

[Paper](./paper.pdf)


## Viewer

The viewer is an interactive GUI fractal renderer.

The viewer uses PyTorch cuda vector code to render fractals efficiently with
only python.

```bash
cd ./viewer

python main.py --alg mandel
python main.py --alg buddha
python main.py --alg nebula
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

### `render/hyprbrot/`

Hyprland themed render of Mandelbrot and Buddhabrot.

# Gallery

![](./gallery/buddhabrot.jpg?raw=true)
![](./gallery/mandelbrot.jpg?raw=true)
![](./gallery/nebulabrot.jpg?raw=true)
![](./gallery/HyprbrotProfile.jpg?raw=true)
