# FNO for the Shallow-Water Equations (1D & 2D)

Train a Fourier Neural Operator (FNO) on shallow-water equation data with
**variable bottom topography**, **f-plane Coriolis** (in 2D) and **periodic
boundary conditions**. Data is generated on the fly with a pseudo-spectral
RK4 solver accelerated on GPU via **CuPy**.

## Equations

**1D**
```
h_t + (h u)_x = 0
u_t + u u_x + g (h + b)_x = 0
```

**2D (f-plane)**
```
h_t + (h u)_x + (h v)_y = 0
u_t + u u_x + v u_y - f v + g (h + b)_x = 0
v_t + u v_x + v v_y + f u + g (h + b)_y = 0
```

`h` is the water depth, `b(x[,y])` is bottom elevation, `f` is the constant
Coriolis parameter (2D only, sampled per trajectory).

## File layout

| file           | role                                                                  |
|----------------|-----------------------------------------------------------------------|
| `config.py`    | dataclasses with all hyperparameters                                  |
| `fno.py`       | `SpectralConv1d/2d`, `FNO1d`, `FNO2d`                                 |
| `solver_1d.py` | pseudo-spectral 1D SWE solver + dataset generator (CuPy RK4 loop)    |
| `solver_2d.py` | pseudo-spectral 2D SWE solver + dataset generator (CuPy RK4 loop)    |
| `cache.py`     | hash-based dataset caching — regenerates only when config changes     |
| `dataset.py`   | `SWEDataset1D`, `SWEDataset2D` (PyTorch `Dataset`s)                  |
| `utils.py`     | relative L² loss, channel-wise normalizer                             |
| `train.py`     | dimension-agnostic train / eval loop                                  |
| `main_1d.py`   | entry point: cache data → train → evaluate (1D)                       |
| `main_2d.py`   | entry point: cache data → train → evaluate (2D)                       |
| `viz_1d.py`    | standalone visualization script for 1D results                        |
| `viz_2d.py`    | standalone visualization script for 2D results                        |

## Operator being learned

* **1D** : `(h₀, u₀, b) ↦ (h(T), u(T))`  — shape `[3, nx] → [2, nx]`
* **2D** : `(h₀, u₀, v₀, b, f) ↦ (h(T), u(T), v(T))` — shape `[5, ny, nx] → [3, ny, nx]`

The Coriolis parameter `f` is constant per sample; it is broadcast to a
constant field so the model can use it as a parameter channel.

## Requirements

```bash
pip install torch numpy cupy-cuda12x matplotlib
```

Replace `cupy-cuda12x` with the variant matching your CUDA version
(e.g. `cupy-cuda11x` for CUDA 11). CuPy is required — the RK4 integration
loop runs exclusively on GPU.

## Usage

### Train

```bash
python main_1d.py     # 1D experiment
python main_2d.py     # 2D experiment
```

On the first run, trajectory data is generated and cached to `data.npz`
alongside a config hash. Subsequent runs with unchanged hyperparameters
load from cache instead of re-running the solver.

To force regeneration, either delete `data.npz` and `data_hash.json` from
the run directory, or change any data-relevant field in `config.py`
(grid resolution, physics constants, dataset sizes, seed).

### Visualize

```bash
python viz_1d.py [--run-dir runs/swe1d]
python viz_2d.py [--run-dir runs/swe2d]
```

Both scripts are standalone — they load the saved checkpoint and data
without re-running training. Figures are written to `{run_dir}/plots/`:

| figure | content |
|--------|---------|
| `training_curves.png` | train/val loss and learning-rate schedule vs epoch |
| `prediction_comparison.png` | truth vs FNO prediction overlaid, per test sample (1D) |
| `error_map.png` | absolute pointwise error per test sample (1D) |
| `field_comparison_sample*.png` | h / u / v truth · pred · error colormaps (2D) |
| `quiver_comparison_sample*.png` | velocity quiver overlaid on depth field (2D) |
| `error_histogram.png` | per-sample relative L² error distribution across test set |

### Outputs (per run directory)

```
runs/swe1d/
    data.npz          # compressed trajectory arrays (train/val/test)
    data_hash.json    # config hash used for cache validation
    fno1d_best.pt     # best checkpoint (weights + normalizer stats)
    history.json      # train/val loss and lr per epoch
    plots/            # PNG figures from viz_1d.py
```

Hyperparameters live in `config.py`. Reduce `n_train`, `epochs`, `nx`, or
`width` for a quick sanity run.

## Design notes

* Periodic BCs are enforced naturally by the pseudo-spectral solver and by
  the FFTs inside the FNO.
* IC generation (random smooth fields, CFL estimate) runs on CPU with NumPy.
  The RK4 integration loop is transferred to GPU via `cp.asarray()` and runs
  entirely in CuPy; results are pulled back with `cp.asnumpy()`.
* The solver applies 2/3-rule dealiasing to the nonlinear products and a
  small `(k²)²` hyperviscosity for stability.
* Trajectories where `h` becomes non-positive or non-finite are rejected
  and re-drawn, so amplitudes can be tuned without crashing the run.
* The trained checkpoint stores both the model weights and the input/output
  normalization statistics, so inference is self-contained.
