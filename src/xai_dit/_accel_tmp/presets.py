"""Model-specific prompt pools and acceleration presets for Phase 1 T2I stress-testing.

Pure Python (no torch/ComfyUI import), so ``--dry-run`` and unit tests run on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    model: str
    label: str
    prompt: str
    negative: str


@dataclass(frozen=True)
class AccelPreset:
    name: str
    stage: str  # "baseline" | "single" | "combo" | "fix" | "fewstep"
    attention: str = "pytorch"  # "pytorch" | "ck" | "flash" | "sage"
    fast_features: tuple[str, ...] = ()  # subset of: fp16_accumulation, fp8_matrix_mult, cublas_ops, autotune
    force_upcast_attention: bool = False
    dont_upcast_attention: bool = False
    easycache: bool = False
    easycache_threshold: float = 0.20
    easycache_start: float = 0.15
    easycache_end: float = 0.95
    torch_compile: Optional[str] = None  # None | "inductor" | "cudagraphs"
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    shift: Optional[float] = None
    dit_override: Optional[str] = None
    lora: Optional[str] = None
    lora_strength: float = 1.0
    zero_negative: bool = False
    notes: str = ""

    def process_flag_key(self) -> tuple[Any, ...]:
        """Process-level ComfyUI flags that must be set before importing comfy.model_management."""
        return (
            self.attention,
            tuple(sorted(self.fast_features)),
            self.force_upcast_attention,
            self.dont_upcast_attention,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fast_features"] = list(self.fast_features)
        return d


ALL_FAST_FEATURES: tuple[str, ...] = (
    "fp16_accumulation",
    "fp8_matrix_mult",
    "cublas_ops",
    "autotune",
)

SAFE_FAST_FEATURES: tuple[str, ...] = (
    "fp8_matrix_mult",
    "cublas_ops",
    "autotune",
)


# --------------------------------------------------------------------------- Prompt pools

ANIMA_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"
MOSS_NEG_2D = "worst quality, bad anatomy, watermark, jpeg artifact, 3d, realistic"
MOSS_NEG_PHOTO = "worst quality, bad anatomy, watermark, jpeg artifact"

PROMPTS_BY_MODEL: dict[str, tuple[PromptSpec, ...]] = {
    "anima": (
        PromptSpec(
            "anima_kanao",
            "anima",
            "@ogipote + tsuyuri kanao",
            "masterpiece, best quality, amazing quality, very aesthetic, absurdres, @ogipote, 1girl, upper body, solo, "
            "expressionless, looking at viewer, tsuyuri kanao, kimetsu no yaiba, demon slayer uniform, sitting, "
            "holding sketchbook, from above, sunlight, soft lighting, futuristic city, glass building, nostalgic, sunset, depth of field",
            ANIMA_NEG,
        ),
        PromptSpec(
            "anima_emilia",
            "anima",
            "emilia + anime coloring",
            "masterpiece, best quality, score_7, safe, 1girl, solo, emilia \\(re:zero\\), re:zero kara hajimeru isekai seikatsu, "
            "upper body, looking at viewer, soft window sunlight, anime coloring, clean lineart",
            ANIMA_NEG,
        ),
    ),
    "moss_image": (
        PromptSpec(
            "moss_emilia",
            "moss_image",
            "emilia 2D (no @artist)",
            "best quality, highres, emilia \\(re:zero\\), re:zero kara hajimeru isekai seikatsu, 1girl, solo, "
            "upper body, looking at viewer, soft window sunlight, anime coloring, clean 2d anime illustration",
            MOSS_NEG_2D,
        ),
        PromptSpec(
            "moss_hoshino",
            "moss_image",
            "hoshino ai 2D (no @artist)",
            "best quality, highres, hoshino ai, oshi no ko, 1girl, solo, star-shaped pupils, rabbit hair ornament, "
            "idol dress, holding a vintage microphone, sunlit rooftop overlooking a coastal city, bright afternoon sky, vivid 2d anime illustration",
            MOSS_NEG_2D,
        ),
        PromptSpec(
            "moss_idol_photo",
            "moss_image",
            "K-pop idol cafe candid (photo)",
            "Candid Instagram portrait photograph of a 23-year-old Korean woman with K-pop idol styling sitting by a sunlit window "
            "in a cozy Seoul cafe, wearing a cream knit cardigan and delicate silver necklace, soft warm afternoon golden-hour rim light "
            "catching her hair strands, gentle fill light on natural skin texture, iced latte on wooden table, shallow depth of field, 50mm f/1.4 lens.",
            MOSS_NEG_PHOTO,
        ),
    ),
    "krea2": (
        PromptSpec(
            "krea2_idol_cafe",
            "krea2",
            "K-pop idol golden-hour cafe (photo)",
            "Candid Instagram portrait photograph of a 23-year-old Korean woman with K-pop idol styling sitting near a floor-to-ceiling window "
            "in a modern Seoul cafe, wearing an ivory ribbed knit cardigan and subtle silver earrings, warm late-afternoon golden-hour sunlight "
            "creating delicate rim lighting along her dark wavy hair and soft directional shadows across her cheek, realistic skin pores and natural makeup, "
            "creamy 50mm f/1.4 bokeh, authentic handheld smartphone-editorial color grading.",
            "",
        ),
        PromptSpec(
            "krea2_idol_neon",
            "krea2",
            "K-pop idol rainy Seoul night (photo)",
            "Candid night street Instagram photograph of a 24-year-old Korean woman with K-pop idol styling standing under a cafe awning on a rainy Seoul street, "
            "wearing a tailored dark navy trench coat and soft red cashmere scarf, warm tungsten shopfront key light illuminating her face while cool cyan and magenta "
            "neon reflections shimmer on wet asphalt behind her, subtle rain droplets in the air, natural skin texture, 35mm lens, shallow depth of field.",
            "",
        ),
        PromptSpec(
            "krea2_emilia",
            "krea2",
            "emilia + anime coloring (verified 2D)",
            "1girl, emilia, re:zero kara hajimeru isekai seikatsu, upper body, looking at viewer, soft natural lighting, anime coloring",
            "",
        ),
    ),
}


def get_prompt(model: str, prompt_id: Optional[str] = None) -> PromptSpec:
    pool = PROMPTS_BY_MODEL[model]
    if prompt_id is None:
        return pool[0]
    for p in pool:
        if p.prompt_id == prompt_id:
            return p
    raise KeyError(f"Unknown prompt_id {prompt_id!r} for model {model!r}; choices: {[p.prompt_id for p in pool]}")


# --------------------------------------------------------------------------- Preset suites

def build_presets_for_model(model: str) -> list[AccelPreset]:
    """Return ordered Phase 1 presets (baseline -> single -> combo -> fix -> fewstep) for a model."""
    presets: list[AccelPreset] = [
        # 0. Baseline (PyTorch SDPA, no fast, no cache)
        AccelPreset("baseline", "baseline", attention="pytorch", notes="Unaccelerated L4 quantized baseline (SDPA)"),
        # 1. Single-factor: Attention backends
        AccelPreset("attn_ck", "single", attention="ck", notes="Comfy Kitchen attention only"),
        AccelPreset("attn_flash", "single", attention="flash", notes="FlashAttention only (if installed)"),
        # SageAttention is tested carefully; explicitly flagged as expected-fail on Krea 2 Raw
        AccelPreset(
            "attn_sage",
            "single",
            attention="sage",
            notes="SageAttention only (expected NaN/black on Krea 2 Raw)",
        ),
        # 2. Single-factor: --fast full vs decomposed PerformanceFeature flags
        AccelPreset("fast_all", "single", fast_features=ALL_FAST_FEATURES, notes="Full --fast (all 4 features)"),
        AccelPreset("fast_cublas", "single", fast_features=("cublas_ops",), notes="--fast cublas_ops only"),
        AccelPreset("fast_fp8mm", "single", fast_features=("fp8_matrix_mult",), notes="--fast fp8_matrix_mult only"),
        AccelPreset("fast_fp16acc", "single", fast_features=("fp16_accumulation",), notes="--fast fp16_accumulation only (graininess suspect)"),
        AccelPreset("fast_no_fp16acc", "fix", fast_features=SAFE_FAST_FEATURES, notes="--fast without fp16_accumulation"),
        # 3. Single-factor: EasyCache
        AccelPreset("ec_t20_s00", "single", easycache=True, easycache_threshold=0.20, easycache_start=0.00, notes="EasyCache from step 0 (early-drift risk)"),
        AccelPreset("ec_t20_s15", "single", easycache=True, easycache_threshold=0.20, easycache_start=0.15, notes="EasyCache default start=0.15"),
        AccelPreset("ec_t20_s25", "fix", easycache=True, easycache_threshold=0.20, easycache_start=0.25, notes="EasyCache delayed start=0.25 (protects early layout)"),
        AccelPreset("ec_t30_s15", "single", easycache=True, easycache_threshold=0.30, easycache_start=0.15, notes="Aggressive EasyCache thresh=0.30"),
        # 4. Single-factor: Sampler comparison
        AccelPreset("sampler_ersde", "single", sampler="er_sde", scheduler="simple", notes="er_sde/simple sampler"),
        AccelPreset("sampler_dpmpp2m", "single", sampler="dpmpp_2m", scheduler="simple", notes="dpmpp_2m/simple sampler"),
        # 5. Combination stress tests (known/suspected collapse & graininess pairs)
        AccelPreset(
            "combo_ck_fast_all",
            "combo",
            attention="ck",
            fast_features=ALL_FAST_FEATURES,
            notes="Step 0 graininess suspect: ck_attention + full --fast",
        ),
        AccelPreset(
            "combo_ec_ersde",
            "combo",
            easycache=True,
            easycache_threshold=0.20,
            easycache_start=0.15,
            sampler="er_sde",
            scheduler="simple",
            notes="Stochastic er_sde + EasyCache interaction test",
        ),
        AccelPreset(
            "combo_ck_fast_all_ec",
            "combo",
            attention="ck",
            fast_features=ALL_FAST_FEATURES,
            easycache=True,
            easycache_threshold=0.20,
            easycache_start=0.15,
            notes="Aggressive 3-way stack: ck + fast(all) + EasyCache",
        ),
        # 6. Candidate fixes (keep speedup, prevent grain/collapse)
        AccelPreset(
            "fix_ck_fast_safe",
            "fix",
            attention="ck",
            fast_features=SAFE_FAST_FEATURES,
            notes="ck_attention + --fast excluding fp16_accumulation",
        ),
        AccelPreset(
            "fix_ck_fast_all_upcast",
            "fix",
            attention="ck",
            fast_features=ALL_FAST_FEATURES,
            force_upcast_attention=True,
            notes="ck_attention + fast(all) + --force-upcast-attention",
        ),
        AccelPreset(
            "fix_full_safe_ec25",
            "fix",
            attention="ck",
            fast_features=SAFE_FAST_FEATURES,
            easycache=True,
            easycache_threshold=0.20,
            easycache_start=0.25,
            sampler="euler",
            notes="Candidate Best Stack: ck + safe --fast + delayed EasyCache(s=0.25) + euler",
        ),
        # 7. Compile benchmark (optional, high warmup cost)
        AccelPreset(
            "compile_inductor",
            "single",
            torch_compile="inductor",
            notes="TorchCompileModel(inductor) — benchmark warmup vs 2nd run",
        ),
    ]

    if model == "krea2":
        presets.extend([
            AccelPreset(
                "krea2_turbo_8step",
                "fewstep",
                attention="pytorch",
                steps=8,
                cfg=1.0,
                sampler="euler",
                scheduler="simple",
                dit_override="krea2_turbo_fp8_scaled.safetensors",
                zero_negative=True,
                notes="Krea 2 Turbo FP8 (8 steps, CFG 1.0, SDPA)",
            ),
            AccelPreset(
                "krea2_turbo_8step_ck_safe",
                "fewstep",
                attention="ck",
                fast_features=SAFE_FAST_FEATURES,
                steps=8,
                cfg=1.0,
                sampler="euler",
                scheduler="simple",
                dit_override="krea2_turbo_fp8_scaled.safetensors",
                zero_negative=True,
                notes="Krea 2 Turbo FP8 + ck_attention + safe --fast",
            ),
        ])

    return presets


def presets_by_name(model: str) -> dict[str, AccelPreset]:
    return {p.name: p for p in build_presets_for_model(model)}
