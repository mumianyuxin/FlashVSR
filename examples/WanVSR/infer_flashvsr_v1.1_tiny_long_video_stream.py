#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import imageio
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from einops import rearrange
from PIL import Image

from diffsynth import ModelManager, FlashVSRTinyLongPipeline
from diffsynth.pipelines.flashvsr_tiny_long import model_fn_wan_video
from utils.TCDecoder import build_tcdecoder
from utils.utils import Causal_LQ4x_Proj


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def compute_scaled_and_target_dims(w0, h0, scale=2.0, multiple=128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple
    if tW == 0 or tH == 0:
        raise ValueError(f"Scaled size too small ({sW}x{sH}) for multiple={multiple}")
    return sW, sH, tW, tH


def upscale_then_center_crop(img, scale, tW, tH):
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    if tW > sW or tH > sH:
        raise ValueError(f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH})")

    up = img.resize((sW, sH), Image.BICUBIC)
    left = (sW - tW) // 2
    top = (sH - tH) // 2
    return up.crop((left, top, left + tW, top + tH))


def pil_to_tensor_neg1_1(img, dtype=torch.bfloat16):
    arr = np.array(img, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).to(dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def tensor_to_uint8_frames_gpu(frames):
    frames = rearrange(frames, "C T H W -> T H W C").contiguous()
    return frames.float().add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8)


@dataclass
class VideoMeta:
    width: int
    height: int
    target_width: int
    target_height: int
    total_frames: int
    padded_frames: int
    output_frames: int
    fps: int


@dataclass
class ChunkProfile:
    read_time: float = 0.0
    lq_proj_time: float = 0.0
    dit_time: float = 0.0
    decode_time: float = 0.0
    color_fix_time: float = 0.0
    write_time: float = 0.0
    total_time: float = 0.0


@dataclass
class RunProfile:
    chunks: list[ChunkProfile] = field(default_factory=list)

    def add(self, chunk: ChunkProfile):
        self.chunks.append(chunk)

    def _avg(self, values):
        return sum(values) / len(values) if values else 0.0

    def summary(self):
        if not self.chunks:
            return {}
        steady = self.chunks[1:] if len(self.chunks) > 1 else []
        source = steady if steady else self.chunks
        return {
            "read": self._avg([c.read_time for c in source]),
            "lq_proj": self._avg([c.lq_proj_time for c in source]),
            "dit": self._avg([c.dit_time for c in source]),
            "decode": self._avg([c.decode_time for c in source]),
            "color_fix": self._avg([c.color_fix_time for c in source]),
            "write": self._avg([c.write_time for c in source]),
            "total": self._avg([c.total_time for c in source]),
        }


@dataclass(frozen=True)
class ChunkPlan:
    stream_slices: list[tuple[int, int]]
    cond_slice: tuple[int, int]
    lq_cur_idx: int
    output_latent_start: int
    output_latent_end: int


class StreamingVideoFrames:
    def __init__(self, path, scale=2.0, dtype=torch.bfloat16, pin_memory=False):
        if not is_video(path):
            raise ValueError(f"Unsupported input: {path}")
        self.path = path
        self.scale = scale
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.reader = imageio.get_reader(path)
        self.cache = {}

        first = Image.fromarray(self.reader.get_data(0)).convert("RGB")
        w0, h0 = first.size
        meta = {}
        try:
            meta = self.reader.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
        total = self._count_frames(meta)
        if total <= 0:
            raise RuntimeError(f"Cannot read frames from {path}")

        _, _, target_w, target_h = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        padded = largest_8n1_leq(total + 4)
        if padded < 25:
            raise RuntimeError(f"Input is too short for tiny-long streaming inference: {total} frames")
        self.meta = VideoMeta(
            width=w0,
            height=h0,
            target_width=target_w,
            target_height=target_h,
            total_frames=total,
            padded_frames=padded,
            output_frames=padded - 4,
            fps=fps,
        )

    def _count_frames(self, meta):
        nf = meta.get("nframes")
        if isinstance(nf, int) and nf > 0 and nf < 10**9:
            return nf
        try:
            return self.reader.count_frames()
        except Exception:
            n = 0
            try:
                while True:
                    self.reader.get_data(n)
                    n += 1
            except Exception:
                return n

    def _read_frame_tensor(self, padded_idx):
        src_idx = min(padded_idx, self.meta.total_frames - 1)
        img = Image.fromarray(self.reader.get_data(src_idx)).convert("RGB")
        img = upscale_then_center_crop(img, self.scale, self.meta.target_width, self.meta.target_height)
        return pil_to_tensor_neg1_1(img, self.dtype)

    def get_slice(self, start, end):
        if start < 0 or end < start:
            raise ValueError(f"Invalid slice [{start}, {end})")
        end = min(end, self.meta.padded_frames)
        frames = []
        for idx in range(start, end):
            if idx not in self.cache:
                frame = self._read_frame_tensor(idx)
                if self.pin_memory:
                    frame = frame.pin_memory()
                self.cache[idx] = frame
            frames.append(self.cache[idx])
        if not frames:
            return None
        out = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        if self.pin_memory and not out.is_pinned():
            out = out.pin_memory()
        return out

    def release_before(self, idx):
        for key in [key for key in self.cache if key < idx]:
            del self.cache[key]

    def close(self):
        try:
            self.reader.close()
        except Exception:
            pass
        self.cache.clear()


def build_chunk_plan(cur_process_idx):
    if cur_process_idx == 0:
        stream_slices = [(max(0, inner_idx * 4 - 3), (inner_idx + 1) * 4 - 3) for inner_idx in range(7)]
        return ChunkPlan(
            stream_slices=stream_slices,
            cond_slice=(0, 21),
            lq_cur_idx=21,
            output_latent_start=0,
            output_latent_end=6,
        )

    base = cur_process_idx * 8
    return ChunkPlan(
        stream_slices=[(base + 17 + inner_idx * 4, base + 21 + inner_idx * 4) for inner_idx in range(2)],
        cond_slice=(base + 13, base + 21),
        lq_cur_idx=base + 21,
        output_latent_start=4 + cur_process_idx * 2,
        output_latent_end=6 + cur_process_idx * 2,
    )


def load_chunk_inputs(reader, plan):
    stream_clips = [reader.get_slice(start, end) for start, end in plan.stream_slices]
    cond = reader.get_slice(*plan.cond_slice)
    reader.release_before(plan.cond_slice[1])
    return {"stream_clips": stream_clips, "cond": cond}


class ChunkPrefetcher:
    def __init__(self, path, scale, dtype, pin_memory=False):
        self.reader = StreamingVideoFrames(path, scale=scale, dtype=dtype, pin_memory=pin_memory)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.chunk_idx = None

    def submit(self, chunk_idx, plan):
        self.future = self.executor.submit(load_chunk_inputs, self.reader, plan)
        self.chunk_idx = chunk_idx

    def pop(self, chunk_idx):
        if self.chunk_idx != chunk_idx or self.future is None:
            return None
        data = self.future.result()
        self.future = None
        self.chunk_idx = None
        return data

    def close(self):
        if self.future is not None:
            self.future.result()
            self.future = None
        self.executor.shutdown(wait=True)
        self.reader.close()


class AsyncVideoWriter:
    def __init__(self, output_path, fps, quality, enabled=True):
        self.writer = imageio.get_writer(output_path, fps=fps, quality=quality)
        self.enabled = enabled
        self.queue = Queue(maxsize=2) if enabled else None
        self.thread = None
        self.stop_token = object()
        self.exception = None

        if self.enabled:
            import threading
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def _worker(self):
        try:
            while True:
                item = self.queue.get()
                if item is self.stop_token:
                    self.queue.task_done()
                    break
                cpu_frames, ready_event = item
                ready_event.synchronize()
                frames_np = cpu_frames.numpy()
                for frame in frames_np:
                    self.writer.append_data(frame)
                self.queue.task_done()
        except Exception as exc:
            self.exception = exc

    def submit(self, cpu_frames, ready_event):
        if self.exception is not None:
            raise self.exception
        if not self.enabled:
            ready_event.synchronize()
            frames_np = cpu_frames.numpy()
            for frame in frames_np:
                self.writer.append_data(frame)
            return
        self.queue.put((cpu_frames, ready_event))

    def close(self):
        if self.enabled:
            self.queue.put(self.stop_token)
            self.thread.join()
        self.writer.close()
        if self.exception is not None:
            raise self.exception


def init_pipeline(device="cuda", dtype=torch.bfloat16, enable_vram_management=False):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    mm.load_models([
        "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
    ])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)

    lq_proj_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(lq_proj_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_proj_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)

    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    missing = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)
    print(missing)

    pipe.to(device)
    if enable_vram_management:
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def concat_lq_latents(prev, cur):
    if cur is None:
        return prev
    if prev is None:
        return cur
    for layer_idx in range(len(prev)):
        prev[layer_idx] = torch.cat([prev[layer_idx], cur[layer_idx]], dim=1)
    return prev


def build_dit_runner(pipe, args):
    def dit_runner(cur_latents, lq_latents, pre_cache_k, pre_cache_v, cur_process_idx, topk_ratio):
        return model_fn_wan_video(
            pipe.dit,
            x=cur_latents,
            timestep=pipe.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=lq_latents,
            is_full_block=False,
            is_stream=True,
            pre_cache_k=pre_cache_k,
            pre_cache_v=pre_cache_v,
            topk_ratio=topk_ratio,
            kv_ratio=args.kv_ratio,
            cur_process_idx=cur_process_idx,
            t_mod=pipe.t_mod,
            t=pipe.t,
            local_range=args.local_range,
        )

    if not args.torch_compile:
        return dit_runner

    compile_kwargs = {}
    if args.torch_compile_mode:
        compile_kwargs["mode"] = args.torch_compile_mode
    if args.torch_compile_backend:
        compile_kwargs["backend"] = args.torch_compile_backend

    print(
        f"[torch.compile] enabled target=model_fn_wan_video "
        f"mode={compile_kwargs.get('mode', 'default')} "
        f"backend={compile_kwargs.get('backend', 'default')}"
    )
    return torch.compile(dit_runner, **compile_kwargs)


def warmup_pipeline(pipe, meta, dit_runner, topk_ratio):
    """Run dummy forward passes to trigger CUDA kernel JIT compilation before the timed loop."""
    device = pipe.device
    dtype = pipe.torch_dtype
    lH = meta.target_height // 8
    lW = meta.target_width // 8

    print(f"[warmup] starting at {meta.target_width}x{meta.target_height} ...")
    t_start = time.perf_counter()
    with torch.inference_mode():
        dummy_lq = torch.zeros(1, 3, 4, meta.target_height, meta.target_width, device=device, dtype=dtype)

        # warm up lq_proj: clip_idx=0 returns None, clips 1-6 run both conv layers
        lq_wm = None
        for _ in range(7):
            cur = pipe.denoising_model().LQ_proj_in.stream_forward(dummy_lq)
            lq_wm = concat_lq_latents(lq_wm, cur)

        # warm up first-chunk DiT (6 latent frames, no pre_cache)
        dummy_x0 = torch.zeros(1, 16, 6, lH, lW, device=device, dtype=dtype)
        wm_k = [None] * len(pipe.dit.blocks)
        wm_v = [None] * len(pipe.dit.blocks)
        _, wm_k, wm_v = dit_runner(dummy_x0, lq_wm, wm_k, wm_v, 0, topk_ratio)

        # warm up steady lq_proj (2 clips)
        lq_wm2 = None
        for _ in range(2):
            cur = pipe.denoising_model().LQ_proj_in.stream_forward(dummy_lq)
            lq_wm2 = concat_lq_latents(lq_wm2, cur)

        # warm up steady DiT (2 latent frames, with populated pre_cache)
        dummy_x1 = torch.zeros(1, 16, 2, lH, lW, device=device, dtype=dtype)
        dit_runner(dummy_x1, lq_wm2, wm_k, wm_v, 1, topk_ratio)

    torch.cuda.synchronize()
    pipe.denoising_model().LQ_proj_in.clear_cache()
    pipe.TCDecoder.clean_mem()
    del dummy_lq, dummy_x0, dummy_x1, wm_k, wm_v, lq_wm, lq_wm2
    torch.cuda.empty_cache()
    print(f"[warmup] done in {time.perf_counter() - t_start:.1f}s")


def run_streaming(args):
    dtype = torch.bfloat16
    use_pinned_memory = args.pin_memory and args.device.startswith("cuda")
    reader = StreamingVideoFrames(args.input, scale=args.scale, dtype=dtype, pin_memory=use_pinned_memory)
    prefetcher = ChunkPrefetcher(args.input, scale=args.scale, dtype=dtype, pin_memory=use_pinned_memory)
    meta = reader.meta
    print(
        f"[{os.path.basename(args.input)}] Original Resolution: {meta.width}x{meta.height} | "
        f"Original Frames: {meta.total_frames} | FPS: {meta.fps}"
    )
    print(
        f"[{os.path.basename(args.input)}] Target Resolution: {meta.target_width}x{meta.target_height} | "
        f"Output Frames: {meta.output_frames} | Padded Frames: {meta.padded_frames}"
    )
    print("[stream] first packet requires ~25 input frames and emits ~21 frames; steady packets emit ~8 frames")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pipe = init_pipeline(args.device, dtype=dtype, enable_vram_management=args.vram_management)

    if hasattr(pipe.dit, "LQ_proj_in"):
        pipe.dit.LQ_proj_in.clear_cache()
    pipe.TCDecoder.clean_mem()

    noise = pipe.generate_noise(
        (1, 16, (meta.padded_frames - 1) // 4, meta.target_height // 8, meta.target_width // 8),
        seed=args.seed,
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    latents = noise
    process_total_num = (meta.padded_frames - 1) // 8 - 2
    topk_ratio = args.sparse_ratio * 768 * 1280 / (meta.target_height * meta.target_width)
    dit_runner = build_dit_runner(pipe, args)

    if not args.skip_warmup:
        warmup_pipeline(pipe, meta, dit_runner, topk_ratio)

    torch.cuda.reset_peak_memory_stats()
    pre_cache_k = None
    pre_cache_v = None
    frames_written = 0
    chunk_times = []
    total_start = time.perf_counter()

    writer = AsyncVideoWriter(
        args.output,
        fps=meta.fps,
        quality=args.quality,
        enabled=not args.disable_async_writer,
    )
    copy_stream = torch.cuda.Stream(device=pipe.device) if args.device.startswith("cuda") else None
    run_profile = RunProfile()
    profiler_table = None
    profiler_trace = None
    try:
        with torch.inference_mode():
            for cur_process_idx in range(process_total_num):
                chunk_start = time.perf_counter()
                chunk_profile = ChunkProfile()
                target_profile = args.torch_profile and cur_process_idx == args.profile_chunk_index
                prof_ctx = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                ) if target_profile else None
                if prof_ctx is not None:
                    prof_ctx.__enter__()
                plan = build_chunk_plan(cur_process_idx)

                t0 = time.perf_counter()
                prefetched = prefetcher.pop(cur_process_idx)
                if prefetched is None:
                    prefetched = load_chunk_inputs(reader, plan)
                chunk_profile.read_time += time.perf_counter() - t0

                next_process_idx = cur_process_idx + 1
                if next_process_idx < process_total_num:
                    prefetcher.submit(next_process_idx, build_chunk_plan(next_process_idx))

                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(pipe.dit.blocks)
                    pre_cache_v = [None] * len(pipe.dit.blocks)
                    lq_latents = None
                    with record_function("flashvsr_lq_proj"):
                        for lq_clip in prefetched["stream_clips"]:
                            t1 = time.perf_counter()
                            cur = pipe.denoising_model().LQ_proj_in.stream_forward(
                                lq_clip.to(pipe.device, non_blocking=use_pinned_memory)
                            )
                            chunk_profile.lq_proj_time += time.perf_counter() - t1
                            lq_latents = concat_lq_latents(lq_latents, cur)
                    cur_latents = latents[:, :, plan.output_latent_start:plan.output_latent_end, :, :]
                else:
                    lq_latents = None
                    with record_function("flashvsr_lq_proj"):
                        for lq_clip in prefetched["stream_clips"]:
                            t1 = time.perf_counter()
                            cur = pipe.denoising_model().LQ_proj_in.stream_forward(
                                lq_clip.to(pipe.device, non_blocking=use_pinned_memory)
                            )
                            chunk_profile.lq_proj_time += time.perf_counter() - t1
                            lq_latents = concat_lq_latents(lq_latents, cur)
                    cur_latents = latents[:, :, plan.output_latent_start:plan.output_latent_end, :, :]

                with record_function("flashvsr_dit"):
                    t0 = time.perf_counter()
                    noise_pred_posi, pre_cache_k, pre_cache_v = dit_runner(
                        cur_latents, lq_latents, pre_cache_k, pre_cache_v, cur_process_idx, topk_ratio
                    )
                    chunk_profile.dit_time += time.perf_counter() - t0

                cur_latents = cur_latents - noise_pred_posi
                cur_lq_frame = prefetched["cond"].to(pipe.device, non_blocking=use_pinned_memory)
                with record_function("flashvsr_decode"):
                    t1 = time.perf_counter()
                    cur_frames = pipe.TCDecoder.decode_video(
                        cur_latents.transpose(1, 2),
                        parallel=False,
                        show_progress_bar=False,
                        cond=cur_lq_frame,
                    ).transpose(1, 2).mul_(2).sub_(1)
                    chunk_profile.decode_time += time.perf_counter() - t1

                if not args.no_color_fix:
                    try:
                        with record_function("flashvsr_color_fix"):
                            t0 = time.perf_counter()
                            cur_frames = pipe.ColorCorrector(
                                cur_frames.to(device=pipe.device),
                                cur_lq_frame,
                                clip_range=(-1, 1),
                                chunk_size=None,
                                method="adain",
                            )
                            chunk_profile.color_fix_time += time.perf_counter() - t0
                    except Exception as exc:
                        print(f"[warning] color_fix failed on chunk {cur_process_idx}: {exc}")

                n_frames = cur_frames.shape[2]
                with record_function("flashvsr_write"):
                    t1 = time.perf_counter()
                    if copy_stream is not None:
                        # Move to_uint8 and DtoH copy onto copy_stream so the main stream is
                        # not blocked by the saturated CUDA kernel queue after DiT+decode.
                        cur_frames_c = cur_frames[0].contiguous()
                        current_stream = torch.cuda.current_stream(pipe.device)
                        copy_stream.wait_stream(current_stream)
                        with torch.cuda.stream(copy_stream):
                            out_frames_gpu = tensor_to_uint8_frames_gpu(cur_frames_c)
                            cpu_frames = torch.empty_like(out_frames_gpu, device="cpu", pin_memory=True)
                            cpu_frames.copy_(out_frames_gpu, non_blocking=True)
                            ready_event = torch.cuda.Event()
                            ready_event.record(copy_stream)
                        writer.submit(cpu_frames, ready_event)
                    else:
                        out_frames_gpu = tensor_to_uint8_frames_gpu(cur_frames[0])
                        cpu_frames = out_frames_gpu.cpu()
                        class _Ready:
                            def synchronize(self_inner):
                                return None
                        writer.submit(cpu_frames, _Ready())
                    chunk_profile.write_time += time.perf_counter() - t1
                frames_written += n_frames

                chunk_time = time.perf_counter() - chunk_start
                chunk_profile.total_time = chunk_time
                run_profile.add(chunk_profile)
                chunk_times.append(chunk_time)
                chunk_fps = n_frames / chunk_time if chunk_time > 0 else float("inf")
                label = "first" if cur_process_idx == 0 else "steady"
                print(
                    f"[stream] chunk={cur_process_idx:04d} type={label} "
                    f"frames={n_frames} input_until={plan.lq_cur_idx} "
                    f"time={chunk_time:.3f}s fps={chunk_fps:.2f} "
                    f"read={chunk_profile.read_time:.3f}s "
                    f"lq={chunk_profile.lq_proj_time:.3f}s "
                    f"dit={chunk_profile.dit_time:.3f}s "
                    f"decode={chunk_profile.decode_time:.3f}s "
                    f"color={chunk_profile.color_fix_time:.3f}s "
                    f"write={chunk_profile.write_time:.3f}s"
                )
                if prof_ctx is not None:
                    torch.cuda.synchronize()
                    prof_ctx.__exit__(None, None, None)
                    profiler_table = prof_ctx.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=args.profile_row_limit,
                    )
                    if args.profile_trace:
                        prof_ctx.export_chrome_trace(args.profile_trace)
                        profiler_trace = args.profile_trace

                del cur_lq_frame, cur_frames, noise_pred_posi, lq_latents
    finally:
        writer.close()
        reader.close()
        prefetcher.close()

    total_time = time.perf_counter() - total_start
    avg_fps = frames_written / total_time if total_time > 0 else 0.0
    steady = chunk_times[1:] if len(chunk_times) > 1 else []
    steady_avg = sum(steady) / len(steady) if steady else 0.0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    stage_summary = run_profile.summary()
    print(
        f"[summary] wrote={frames_written} frames total_time={total_time:.3f}s "
        f"avg_fps={avg_fps:.2f} first_latency={chunk_times[0]:.3f}s "
        f"steady_chunk_avg={steady_avg:.3f}s peak_mem={peak_gb:.2f}GiB"
    )
    if stage_summary:
        print(
            f"[summary.steady] read={stage_summary['read']:.3f}s "
            f"lq={stage_summary['lq_proj']:.3f}s "
            f"dit={stage_summary['dit']:.3f}s "
            f"decode={stage_summary['decode']:.3f}s "
            f"color={stage_summary['color_fix']:.3f}s "
            f"write={stage_summary['write']:.3f}s "
            f"total={stage_summary['total']:.3f}s"
        )
    if profiler_table:
        print("[torch.profiler]")
        print(profiler_table)
        if profiler_trace:
            print(f"[torch.profiler.trace] {profiler_trace}")
    return args.output


def mse_to_psnr(mse):
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def compare_psnr(baseline_path, output_path, threshold):
    base = imageio.get_reader(baseline_path)
    out = imageio.get_reader(output_path)
    psnrs = []
    min_psnr = float("inf")
    min_idx = -1
    idx = 0
    try:
        while True:
            try:
                base_frame = base.get_data(idx)
            except Exception:
                try:
                    out.get_data(idx)
                    raise RuntimeError(f"Output has more frames than baseline at frame {idx}")
                except IndexError:
                    break
                except RuntimeError:
                    raise
                except Exception:
                    break
            try:
                out_frame = out.get_data(idx)
            except Exception as exc:
                raise RuntimeError(f"Output ended before baseline at frame {idx}") from exc
            if base_frame.shape != out_frame.shape:
                raise RuntimeError(f"Frame {idx} shape mismatch: baseline={base_frame.shape}, output={out_frame.shape}")
            diff = base_frame.astype(np.float32) - out_frame.astype(np.float32)
            psnr = mse_to_psnr(float(np.mean(diff * diff)))
            psnrs.append(psnr)
            if min_idx < 0 or psnr < min_psnr:
                min_psnr = psnr
                min_idx = idx
            idx += 1
    finally:
        base.close()
        out.close()

    if not psnrs:
        raise RuntimeError("No frames compared")
    avg_psnr = sum(psnrs) / len(psnrs)
    boundary = [20, 21] + [21 + 8 * i for i in range(max(0, (len(psnrs) - 21 + 7) // 8))]
    boundary = [i for i in boundary if 0 <= i < len(psnrs)]
    boundary_min = min((psnrs[i], i) for i in boundary) if boundary else (float("nan"), -1)
    print(
        f"[psnr] frames={len(psnrs)} avg={avg_psnr:.3f}dB "
        f"min={min_psnr:.3f}dB@{min_idx} boundary_min={boundary_min[0]:.3f}dB@{boundary_min[1]} "
        f"threshold={threshold:.3f}dB"
    )
    if avg_psnr < threshold:
        raise RuntimeError(f"Average PSNR {avg_psnr:.3f}dB is below threshold {threshold:.3f}dB")


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming FlashVSR v1.1 tiny-long inference")
    parser.add_argument("--input", default="./inputs/example0.mp4")
    parser.add_argument("--output", default="./results/FlashVSR_v1.1_Tiny_Long_stream.mp4")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-ratio", type=float, default=2.0)
    parser.add_argument("--kv-ratio", type=float, default=3.0)
    parser.add_argument("--local-range", type=int, default=11)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-color-fix", action="store_true")
    parser.add_argument("--vram-management", action="store_true")
    parser.add_argument("--compare-baseline", default=None)
    parser.add_argument("--psnr-threshold", type=float, default=40.0)
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--profile-chunk-index", type=int, default=1)
    parser.add_argument("--profile-row-limit", type=int, default=30)
    parser.add_argument("--profile-trace", default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="max-autotune")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--disable-async-writer", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="skip CUDA warmup pass (useful to measure raw JIT overhead)")
    return parser.parse_args()


def main():
    args = parse_args()
    output = run_streaming(args)
    if args.compare_baseline:
        compare_psnr(args.compare_baseline, output, args.psnr_threshold)
    print("Done.")


if __name__ == "__main__":
    main()
