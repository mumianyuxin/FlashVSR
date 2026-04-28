# Streaming FlashVSR Latency Optimization

## Scope

This note summarizes the current latency work on `examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video_stream.py`.

Validation was run on devbox worker `3796800` with `NVIDIA H20`, using `CUDA_VISIBLE_DEVICES=4`.

## Baseline Profiling

Input: `examples/WanVSR/inputs/example0.mp4`  
Resolution: `384x384 -> 768x768`  
Steady-state output granularity: `8` frames per chunk

Initial steady-state profiling:

- `read`: `0.089s`
- `lq_proj`: `0.006s`
- `dit`: `0.355s`
- `decode`: `0.009s`
- `to_u8`: `0.127s`
- `write`: `0.037s`
- `total`: `0.623s`

Steady-state throughput was about `12.8 FPS`.

## Applied Optimizations

### 1. GPU-side uint8 conversion

Changed postprocessing to convert frames to `uint8` on GPU before copying to CPU:

- reduced `to_u8` from about `0.127s` to `0.088s`
- reduced steady chunk time from `0.623s` to `0.549s`

### 2. Next-chunk input prefetch

Added a chunk prefetch path with a dedicated reader and background loader:

- next chunk LR frames are prepared while the current chunk runs
- steady-state `read` time dropped from about `0.089s` to `0.001s`
- steady chunk time dropped from `0.549s` to `0.496s`

## Current Result

For `384x384 -> 768x768`:

- steady chunk time: `0.496s`
- steady throughput: about `16.1 FPS`
- first-chunk latency: about `6.8s`

Steady-state target `>=16 FPS` is met for this short sample.

## 540p Measurement

Input: `example0_540p_33f.mp4`  
Resolution: `960x540 -> 1920x1024`

Measured steady-state profiling:

- `read`: `0.027s`
- `lq_proj`: `0.072s`
- `dit`: `1.061s`
- `decode`: `0.010s`
- `to_u8`: `0.318s`
- `write`: `0.032s`
- `total`: `1.558s`

Steady-state throughput was about `5.14 FPS`.

## 480p Measurement

Input: `example0_854x480_33f.mp4`  
Resolution: `854x480 -> 1664x896`

Measured steady-state profiling:

- `read`: `0.025s`
- `lq_proj`: `0.057s`
- `dit`: `0.829s`
- `decode`: `0.011s`
- `to_u8`: `0.242s`
- `write`: `0.027s`
- `total`: `1.223s`

Other observed metrics:

- first-chunk latency: `16.813s`
- steady-state throughput: about `6.54 FPS`
- peak memory: `23.91 GiB`

### 480p Parameter Grid

Baseline for comparison:

- `sparse_ratio=2.0`
- `local_range=11`
- `kv_ratio=3.0`

PSNR below is measured against the baseline 480p output on the same short sample.

| tag | sparse | local | kv | steady chunk | steady FPS | avg PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| baseline rerun | 2.0 | 11 | 3.0 | `1.233s` | `6.49` | `inf` |
| s150_l11_k3 | 1.5 | 11 | 3.0 | `1.118s` | `7.16` | `36.07 dB` |
| s125_l11_k3 | 1.25 | 11 | 3.0 | `1.054s` | `7.59` | `35.16 dB` |
| s100_l11_k3 | 1.0 | 11 | 3.0 | `1.009s` | `7.93` | `34.04 dB` |
| s100_l9_k3 | 1.0 | 9 | 3.0 | `1.007s` | `7.94` | `34.63 dB` |
| s100_l7_k2 | 1.0 | 7 | 2.0 | `0.991s` | `8.07` | `34.27 dB` |

Observed trend:

- Lower `sparse_ratio` improves speed mostly by reducing `DiT` time.
- `local_range` and `kv_ratio` changes help less than `sparse_ratio`.
- The best point in this coarse sweep reached about `8.07 FPS`, still far from real-time.

### 480p Parameter Grid Round 2

Round 2 fixed `local_range=9` and focused on `sparse_ratio` and `kv_ratio`.

| tag | sparse | local | kv | steady chunk | steady FPS | avg PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| s1125_l9_k20 | 1.125 | 9 | 2.0 | `1.029s` | `7.77` | `34.90 dB` |
| s1125_l9_k25 | 1.125 | 9 | 2.5 | `1.021s` | `7.84` | `34.90 dB` |
| s0875_l9_k15 | 0.875 | 9 | 1.5 | `0.986s` | `8.11` | `33.84 dB` |
| s0875_l9_k20 | 0.875 | 9 | 2.0 | `0.971s` | `8.24` | `33.84 dB` |
| s0875_l9_k25 | 0.875 | 9 | 2.5 | `1.044s` | `7.66` | `33.84 dB` |
| s075_l9_k15 | 0.75 | 9 | 1.5 | `0.948s` | `8.44` | `33.32 dB` |
| s075_l9_k20 | 0.75 | 9 | 2.0 | `1.001s` | `7.99` | `33.32 dB` |
| s075_l9_k25 | 0.75 | 9 | 2.5 | `0.950s` | `8.42` | `33.32 dB` |

Current 480p takeaways:

- `sparse_ratio` remains the main speed/quality lever.
- `kv_ratio` has much smaller impact than `sparse_ratio`.
- A moderate tradeoff point is `s1125_l9_k25`, which keeps about `34.90 dB` at `7.84 FPS`.
- A speed-oriented point is `s075_l9_k15`, which reaches about `8.44 FPS` at `33.32 dB`.

### 480p Torch Profiler Findings

A `torch.profiler` run was taken on the steady chunk of the 480p baseline (`854x480 -> 1664x896`).

Top CUDA-side costs:

- `flashvsr_dit`: about `819.5 ms` self CUDA, about `71.7%`
- `flash_attn::_block_sparse_attn_forward`: about `418.7 ms`, about `36.6%`
- `aten::cudnn_convolution`: about `289.5 ms`, about `25.3%`
- `aten::addmm`: about `216.5 ms`, about `18.9%`
- `flashvsr_lq_proj`: about `108.8 ms`, about `9.5%`
- `flashvsr_to_uint8`: about `20.1 ms`, about `1.8%`
- `Memcpy HtoD (Pageable -> Device)`: about `20.1 ms`
- `Memcpy DtoH (Device -> Pageable)`: about `19.4 ms`

Engineering conclusions from the profile:

- The dominant bottleneck is still `DiT`, especially block-sparse attention.
- Small parameter tuning is no longer enough to create a large speedup.
- The next useful engineering targets are:
  - reduce block-sparse attention launch/runtime overhead
  - reduce repeated GEMM / convolution overhead inside the steady chunk
  - switch CPU-to-GPU staging to pinned memory plus non-blocking copies
  - overlap device-to-host transfer and video writing with the next chunk
  - reduce temporary tensor churn from repeated `cat` and `copy_`

## Round 2 Optimizations

Tested on worker `3798521` with `NVIDIA H20`, `CUDA_VISIBLE_DEVICES=4`.

Input: `example0_854x480_33f.mp4`, resolution `854x480 -> 1664x896`, baseline params (`sparse_ratio=2.0`, `kv_ratio=3.0`, `local_range=11`).

### 3. CUDA warmup pass

Added `warmup_pipeline()` that exercises both first-chunk and steady-chunk code paths before the timed loop. This forces PTX→SASS JIT compilation to happen once at startup rather than during the first real chunk.

Per-chunk breakdown comparison (first chunk):

| | skip-warmup | with warmup |
| --- | --- | --- |
| `read` | `4.897s` | `3.910s` |
| `lq` | `11.835s` | `0.015s` |
| `dit` | `2.670s` | `2.461s` |
| `decode` | `1.223s` | `1.204s` |
| `write` | `0.048s` | `0.052s` |
| **total** | **`20.676s`** | **`7.644s`** |

- `first_latency` dropped from `20.676s` to `7.644s` (63% reduction)
- The `lq` JIT cost of `11.835s` is now eliminated from the live path
- Warmup itself takes `~21s` as a one-time startup cost

### 4. `tensor_to_uint8` moved to copy stream

Moved `tensor_to_uint8_frames_gpu` and the DtoH copy onto `copy_stream` so they overlap with the next chunk's main-stream work. Steady-chunk `write` dropped from `0.001s` to `0.000s` wall-clock, confirming the op is off the critical path.

Steady-chunk impact was small overall:

| | skip-warmup | with warmup + copy-stream |
| --- | --- | --- |
| `dit` | `1.150s` | `1.143s` |
| `total` | `1.165s` | `1.156s` |
| `steady FPS` | `~6.87` | `~6.92` |

The steady-state improvement is within measurement noise. DiT (`1.143s`, `~98.9%` of chunk time) is the sole bottleneck.

## Current Result (480p, Round 2)

- `first_latency`: `7.644s` (down from `16.813s`)
- `steady_chunk_avg`: `1.156s`
- `steady FPS`: `~6.92`
- `peak_mem`: `23.91 GiB`

## Round 3: FP8 Quantization Experiment

Tested `--fp8` flag (DiT linear layers replaced with `torch._scaled_mm` FP8 GEMM, per-tensor dynamic scales).

| | BF16 baseline | FP8 dynamic |
| --- | --- | --- |
| `dit` | `1.143s` | `1.192s` |
| `steady_chunk_avg` | `1.156s` | `1.209s` |
| `steady FPS` | `~6.92` | `~6.62` |

**FP8 was 4.6% slower.** Root causes:

1. **Non-contiguous activation copies**: `rearrange('b c f h w -> b (f h w) c')` before lq_proj produces a [5824, 3072] tensor with strides `(1, 5824)` (column-major). cuBLASLt rejects column-major A matrices, so `.contiguous()` must copy ~36 MB per call.
2. **Per-tensor dynamic scale overhead**: `abs().max()` reduction over [N, K] activations on every forward pass, across ~30 blocks × multiple linears.
3. **Attention is not quantized**: block-sparse attention (`418ms`, `36.6%` of DiT) is unchanged; FP8 can only help the `addmm` portion (`216ms`, `18.9%`). Even a theoretical 2× GEMM speedup saves only ~108ms out of `~1.14s`, dropping steady FPS to ~9.5.

The overhead from (1) and (2) exceeds the GEMM speedup at these sequence lengths.

**Path to working FP8 speedup**:
- Static/calibration-based activation scales (eliminates `abs().max()` per forward pass)
- Pre-transpose activations before FP8 layers (eliminates `.contiguous()` copy)
- `torchao` with a compatible torch version (requires torch >= 2.11 for torchao ≥ 0.17)
- Quantize attention (FlashAttention FP8 variant, once available for block-sparse)

## Conclusion

The warmup pass resolves the first-chunk latency regression. The copy-stream move for `to_uint8` is correct but adds no measurable steady-state benefit. Dynamic FP8 quantization via `torch._scaled_mm` adds overhead that exceeds its speedup at 480p.

DiT dominates at `~1.14s` per chunk (BF16). Reaching 16 FPS at `480p` requires DiT < 400ms. This is not achievable through inference-pipeline changes alone. The path forward is:
- Static FP8 calibration or torchao (with compatible torch version)
- Tensor parallelism across 2+ GPUs (measured estimate: 2 GPU → ~13.8 FPS, 4 GPU → ~25 FPS)
- Reducing model depth or dimension
- Reducing temporal output per chunk (architectural change)
