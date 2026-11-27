"""Pure-Python CPU unit tests for Phase 3 acceleration profiler, probes, and extended runner.

Zero ``torch`` dependency — verifies syntax compilation, TimingBreakdown math,
CUDA kernel categorization, Scheduled CFG schedule math, tier/track preset filtering,
optional custom-node detection, and CLI ``--dry-run`` paths.
"""

from __future__ import annotations

import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dit._accel_tmp.phase3_extended import (  # noqa: E402
    build_extended_presets,
    compute_cfg_schedule_plan,
    compute_scheduled_cfg_value,
    detect_custom_nodes,
    main as extended_main,
)
from xai_dit._accel_tmp.phase3_probes import (  # noqa: E402
    PROBE_IDS,
    main as probes_main,
    select_probe_jobs,
)
from xai_dit._accel_tmp.phase3_profiler import (  # noqa: E402
    TimingBreakdown,
    categorize_kernel,
    cuda_timer,
    summarize_kernel_events,
)


class TestPhase3PyCompile(unittest.TestCase):
    def test_all_phase3_scripts_compile(self) -> None:
        accel_dir = SRC_DIR / "xai_dit" / "_accel_tmp"
        for fname in ("phase3_profiler.py", "phase3_probes.py", "phase3_extended.py"):
            path = accel_dir / fname
            self.assertTrue(path.is_file(), f"Missing {path}")
            py_compile.compile(str(path), doraise=True)


class TestPhase3Profiler(unittest.TestCase):
    def test_timing_breakdown_derived_metrics(self) -> None:
        tb = TimingBreakdown.from_measurements(
            t_te_encode_s=1.25,
            t_compile_warmup_s=45.0,
            t_denoise_pure_s=18.0,
            actual_forwards=36,
            nominal_forwards=60,
            t_vae_decode_s=0.45,
            vram_peak_gb=14.2,
            vram_reserved_gb=16.0,
        )
        self.assertAlmostEqual(tb.s_per_fwd, 0.5, places=4)
        self.assertAlmostEqual(tb.skip_ratio, 0.4, places=4)
        d = tb.to_dict()
        self.assertEqual(d["actual_forwards"], 36)
        self.assertEqual(d["nominal_forwards"], 60)
        self.assertEqual(d["vram_peak_gb"], 14.2)
        self.assertEqual(d["vram_reserved_gb"], 16.0)

    def test_timing_breakdown_zero_guards(self) -> None:
        tb = TimingBreakdown.from_measurements(
            t_denoise_pure_s=0.0,
            actual_forwards=0,
            nominal_forwards=0,
        )
        self.assertEqual(tb.s_per_fwd, 0.0)
        self.assertEqual(tb.skip_ratio, 0.0)

    def test_cuda_timer_cpu_fallback(self) -> None:
        with cuda_timer(torch_mod=None) as timer:
            _ = sum(range(1000))
        self.assertGreaterEqual(timer.elapsed_s, 0.0)
        self.assertFalse(timer.used_cuda_events)

    def test_kernel_categorization(self) -> None:
        self.assertEqual(categorize_kernel("comfy_kitchen::_int8_mm_cuda"), "int8_gemm")
        self.assertEqual(categorize_kernel("cutlass_80_tensorop_i8_gemm_128x128"), "int8_gemm")
        self.assertEqual(categorize_kernel("cublasLt_fp8_e4m3_scaled_mm"), "fp8_gemm")
        self.assertEqual(categorize_kernel("aten::_scaled_mm"), "fp8_gemm")
        self.assertEqual(categorize_kernel("nvjet_tst_128x128_64x4_bf16_gemm"), "bf16_gemm")
        self.assertEqual(categorize_kernel("ampere_bf16_s16816gemm_bf16"), "bf16_gemm")
        self.assertEqual(categorize_kernel("flash_fwd_splitkv_kernel"), "attn")
        self.assertEqual(categorize_kernel("sageattn_int8_pv_fp16"), "attn")
        self.assertEqual(categorize_kernel("ck::sol_attn_forward"), "attn")
        self.assertEqual(categorize_kernel("aten::_to_copy"), "dtype_cast")
        self.assertEqual(categorize_kernel("aten::copy_"), "dtype_cast")
        self.assertEqual(categorize_kernel("vectorized_elementwise_silu_kernel"), "other")

    def test_summarize_kernel_events(self) -> None:
        events = [
            {"name": "aten::_to_copy", "cuda_ms": 2.4, "calls": 168},
            {"name": "comfy_kitchen::_int8_mm", "cuda_ms": 12.5, "calls": 40},
            {"name": "flash_fwd_kernel", "cuda_ms": 5.1, "calls": 28},
            {"name": "nvjet_bf16_gemm", "cuda_ms": 1.0, "calls": 4},
        ]
        summary = summarize_kernel_events(events, top_k=2)
        self.assertEqual(len(summary["top_kernels"]), 2)
        self.assertEqual(summary["top_kernels"][0]["category"], "int8_gemm")
        self.assertEqual(summary["dtype_cast_calls"], 168)
        self.assertTrue(summary["has_int8_gemm"])
        self.assertFalse(summary["has_fp8_gemm"])
        self.assertTrue(summary["has_bf16_gemm"])
        self.assertAlmostEqual(summary["total_cuda_ms"], 21.0, places=3)


class TestPhase3ExtendedAndProbes(unittest.TestCase):
    def test_scheduled_cfg_calculation(self) -> None:
        # 30 steps, base CFG 4.0, drop at 60% -> steps 0..17 use 4.0 (18 steps), steps 18..29 use 1.0 (12 steps)
        self.assertEqual(compute_scheduled_cfg_value(0, 30, 4.0, 0.60), 4.0)
        self.assertEqual(compute_scheduled_cfg_value(17, 30, 4.0, 0.60), 4.0)
        self.assertEqual(compute_scheduled_cfg_value(18, 30, 4.0, 0.60), 1.0)
        self.assertEqual(compute_scheduled_cfg_value(29, 30, 4.0, 0.60), 1.0)

        plan = compute_cfg_schedule_plan(30, 4.0, 0.60)
        self.assertEqual(plan["nominal_forwards"], 60)
        # 18 * 2 + 12 * 1 = 48 forwards -> skip_ratio = 1 - 48/60 = 0.20
        self.assertEqual(plan["expected_forwards"], 48)
        self.assertAlmostEqual(plan["expected_skip_ratio"], 0.20, places=4)

    def test_tier_preset_filtering(self) -> None:
        for model in ("anima", "moss_image", "krea2", "krea2_anime"):
            l4_presets = build_extended_presets(model, tier="l4", track="all")
            self.assertTrue(len(l4_presets) >= 2, f"Expected L4 presets for {model}")
            self.assertTrue(all(p.tier == "l4" for p in l4_presets), f"Blackwell preset leaked into L4 for {model}")

            bw_presets = build_extended_presets(model, tier="blackwell", track="all")
            self.assertTrue(any(p.tier == "blackwell" for p in bw_presets), f"No Blackwell presets for {model}")

        # Verify track filtering always prepends baseline for comparison
        anima_cfg = build_extended_presets("anima", tier="l4", track="cfg_attn")
        self.assertEqual(anima_cfg[0].name, "baseline")
        self.assertTrue(any(p.cfg_drop_percent == 0.60 for p in anima_cfg))
        self.assertTrue(any(p.block_sparse_tau == 1.3 for p in anima_cfg))

        # Verify Krea2-Anime v3.0 quantization formats
        ka_quant = build_extended_presets("krea2_anime", tier="l4", track="quant")
        names = {p.name for p in ka_quant}
        self.assertIn("quant_krea2_anime_int8_convrot", names)
        self.assertIn("quant_krea2_anime_int4_w4a4", names)

    def test_detect_custom_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cn = Path(tmp) / "custom_nodes"
            (cn / "ComfyUI-Spectrum-KSampler").mkdir(parents=True)
            (cn / "comfyui-moss-accel").mkdir(parents=True)
            detected = detect_custom_nodes(tmp)
            self.assertIsNotNone(detected["spectrum_ksampler"])
            self.assertIsNotNone(detected["moss_accel"])
            self.assertIsNone(detected["anima_teacache"])

    def test_probes_registry_and_dry_run(self) -> None:
        all_jobs = select_probe_jobs("all")
        seen_probes = {j["probe"] for j in all_jobs}
        self.assertEqual(seen_probes, set(PROBE_IDS))

        with tempfile.TemporaryDirectory() as tmp:
            rc1 = probes_main(["--probe", "all", "--out-dir", tmp, "--dry-run"])
            self.assertEqual(rc1, 0)
            rc2 = extended_main(["--tier", "l4", "--model", "anima", "--track", "cfg_attn", "--out-dir", tmp, "--dry-run"])
            self.assertEqual(rc2, 0)


if __name__ == "__main__":
    unittest.main()
