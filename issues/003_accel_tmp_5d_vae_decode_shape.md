# ISSUE-003: `_accel_tmp` 5D VAE Decode Output Shape & Hidden Subprocess Traceback

- **Date**: 2026-11-26
- **Status**: Resolved
- **Model / Module**: `anima` (and `krea2`, Wan2.1 5D VAE `qwen_image_vae.safetensors`); `src/xai_dit/_accel_tmp/run_t2i.py`, `src/xai_dit/_accel_tmp/metrics_and_exif.py`, `src/xai_dit/_accel_tmp/sweep.py`
- **Related Experiments**: Phase 1 acceleration sweep (`anima`, `anima_kanao`, seed `42`, Colab L4 `xai-l4`)

---

## 1. Problem

The first Colab L4 run of `src/xai_dit/_accel_tmp/sweep.py` on `anima` (`anima_kanao`, seed `42`) failed across all 19 presets with `subprocess exit code 1`, and `sweep.py` did not print any traceback to identify the failure.

---

## 2. Observation & Evidence

Running `src/xai_dit/_accel_tmp/run_t2i.py` directly on the `baseline` preset captured the underlying failure:

- Text encoder encoding (`qwen_3_06b_base.safetensors`) succeeded in `3.6s`.
- DiT (`anima-base-v1.0.safetensors`) loaded and sampled cleanly.
- Image saving raised:
  ```text
  TypeError: Cannot handle this data type: (1, 1, 1024, 3), |u1
  ```
  at `PIL.Image.fromarray(rgb_uint8)` in `src/xai_dit/_accel_tmp/metrics_and_exif.py:207` (invoked from `run_t2i.py:483` prior to the fix, now `run_t2i.py:486`).

---

## 3. Root Cause / Hypothesis

1. **5D VAE Decode Shape (`run_t2i.py`)**: Anima (and Krea 2) use the Wan2.1 5D VAE (`qwen_image_vae.safetensors`), whose `vae.decode(samples)` returns a 5D tensor `[B, T, H, W, 3] = [1, 1, 1024, 1024, 3]`. Indexing only `decoded[0]` left a 4D array `(1, 1024, 1024, 3)` instead of a 3D `(1024, 1024, 3)` RGB array, causing `PIL.Image.fromarray(rgb_uint8)` to raise `TypeError`.
2. **Suppressed Subprocess Output (`sweep.py`)**: `sweep.py` invoked `subprocess.run` without capturing and printing `stderr`, hiding the Python traceback during multi-preset sweep runs.

---

## 4. Fix / Method Applied

1. **`src/xai_dit/_accel_tmp/run_t2i.py` (L403–L406)**: Replaced single `decoded[0]` indexing with a dimension-squeezing loop:
   ```python
   img_t = decoded
   while img_t.ndim > 3:
       img_t = img_t[0]
   ```
   This normalizes both 4D (`[1, H, W, 3]`, MOSS-Image Flux2 VAE) and 5D (`[1, 1, H, W, 3]`, Anima / Krea 2 Wan2.1 VAE) decode outputs to `(H, W, 3)`.
2. **`src/xai_dit/_accel_tmp/sweep.py` (L168–L174)**: Updated `subprocess.run` to use `capture_output=True, text=True`, printing `proc.stdout` and trailing `proc.stderr[-1500:]` on non-zero exit codes.

---

## 5. Verification & Improvement

- **Colab L4 Re-Run (`xai-l4`, `anima`, `anima_kanao`, seed `42`)**: Re-ran the Anima Phase 1 19-preset sweep after uploading the fixed `run_t2i.py` and `sweep.py`.
- **Result**: **19/19 OK** (`0` errors, `0` collapses) — all 19 presets decoded and saved PNGs cleanly without shape errors.
