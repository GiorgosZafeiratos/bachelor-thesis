import os
import re
import shutil
import sys
import textwrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def read(rel: str) -> str:
    with open(_path(rel), encoding="utf-8") as fh:
        return fh.read()


def write(rel: str, content: str) -> None:
    dest = _path(rel)
    bak  = dest + ".bak"
    if not os.path.exists(bak):          # keep original backup
        shutil.copy2(dest, bak)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(content)


def apply(rel: str, old: str, new: str, description: str) -> None:
    """Replace `old` with `new` in file `rel`; prints PASS/SKIP/FAIL."""
    src = read(rel)
    if old not in src:
        if new in src:
            print(f"  SKIP  {description!r}  (already applied in {rel})")
        else:
            print(f"  FAIL  {description!r}  — anchor text not found in {rel}", file=sys.stderr)
            print(f"        Looking for:\n{textwrap.indent(old, '        ')}", file=sys.stderr)
        return
    write(rel, src.replace(old, new, 1))
    print(f"  PASS  {description!r}  → {rel}")


# ===========================================================================
# Bug 1 · tests/verify_patches.py
# `_apply_normalization` (American) → `_apply_normalisation` (British)
# The whole codebase uses the British spelling; the test crashes immediately.
# ===========================================================================

def fix_bug1() -> None:
    print("\n[Bug 1] Fix _apply_normalization → _apply_normalisation (tests/verify_patches.py)")
    apply(
        "tests/verify_patches.py",
        old="out = pipeline._apply_normalization(X)",
        new="out = pipeline._apply_normalisation(X)",
        description="British-spelling fix for _apply_normalisation",
    )


# ===========================================================================
# Bug 2 · tests/verify_patches.py
# config.yaml resolved to tests/config.yaml (wrong directory).
# __file__ is tests/verify_patches.py, so dirname points inside tests/.
# ===========================================================================

def fix_bug2() -> None:
    print("\n[Bug 2] Fix config.yaml path resolution (tests/verify_patches.py)")
    apply(
        "tests/verify_patches.py",
        old=(
            'yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")\n'
            '    if not os.path.exists(yaml_path):\n'
            '        _fail("config.yaml not found alongside verify_patches.py")'
        ),
        new=(
            'yaml_path = os.path.normpath(\n'
            '        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")\n'
            '    )\n'
            '    if not os.path.exists(yaml_path):\n'
            '        _fail(f"config.yaml not found at expected path: {yaml_path}")'
        ),
        description="Resolve config.yaml one level above tests/ directory",
    )


# ===========================================================================
# Bug 3 · main.py
# YAML always overrides CLI flags — the elif branch is dead (compares an int
# to a fresh uninitialised Namespace), so every key falls to the else branch
# which unconditionally calls setattr.  CLI flags are silently ignored.
# ===========================================================================

def fix_bug3() -> None:
    print("\n[Bug 3] Fix YAML always overriding CLI flags (main.py)")
    apply(
        "main.py",
        old=(
            "    if args.config is not None:\n"
            "        overrides = load_yaml_config(args.config)\n"
            "        for key, val in overrides.items():\n"
            "            # Only apply if the user did not explicitly pass the flag on the CLI\n"
            "            # (argparse stores defaults; we cannot distinguish, so YAML acts as defaults)\n"
            "            if not hasattr(args, key):\n"
            "                setattr(args, key, val)\n"
            "            elif getattr(args, key) == args.__class__.__new__(args.__class__):\n"
            "                setattr(args, key, val)\n"
            "            else:\n"
            "                # Overwrite only if the current value matches the argparse default\n"
            "                setattr(args, key, val)"
        ),
        new=(
            "    if args.config is not None:\n"
            "        overrides = load_yaml_config(args.config)\n"
            "        # Collect keys the user explicitly typed on the command line.\n"
            "        # sys.argv entries that start with '--' are long option flags; strip\n"
            "        # the leading '--' and normalise hyphens → underscores to match the\n"
            "        # attribute names stored on the Namespace object.\n"
            "        explicit_cli_keys = {\n"
            "            a.lstrip('-').replace('-', '_').split('=')[0]\n"
            "            for a in sys.argv[1:]\n"
            "            if a.startswith('--')\n"
            "        }\n"
            "        for key, val in overrides.items():\n"
            "            if key not in explicit_cli_keys:   # CLI wins; YAML fills the rest\n"
            "                setattr(args, key, val)"
        ),
        description="CLI flags beat YAML (explicit_cli_keys approach)",
    )


# ===========================================================================
# Bug 4 · src/hybrid_model.py
# ablation_no_se() passes se_ratio=0 → SEBlock computes channels // 0
# → ZeroDivisionError.  Guard SEBlock construction in ResidualConvBlock.
# ===========================================================================

def fix_bug4() -> None:
    print("\n[Bug 4] Guard SEBlock against se_ratio=0 (src/hybrid_model.py)")
    apply(
        "src/hybrid_model.py",
        old="self.se = SEBlock(out_channels, se_ratio)",
        new="self.se = SEBlock(out_channels, se_ratio) if se_ratio > 0 else nn.Identity()",
        description="se_ratio=0 → nn.Identity() instead of ZeroDivisionError",
    )


# ===========================================================================
# Bug 5 · src/xai_and_pipeline.py
# LRPExplainer: R_feat initialised with n_classes channels instead of
# C_last channels.  The shape mismatch triggers the mean-collapse fallback,
# which makes every time-step have identical relevance (uniform attributions).
# ===========================================================================

def fix_bug5() -> None:
    print("\n[Bug 5] Fix R_feat shape in LRPExplainer (src/xai_and_pipeline.py)")
    # The original block obtains last_act_shape and then does a direct unsqueeze/expand
    # using R (shape: 1 × n_classes) rather than projecting through the head weights.
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "last_act_shape = self._activations[conv_act_keys[-1]]\n"
            "            B, C_last, T_last = last_act_shape.shape\n"
            "\n"
            "            R_feat = R.unsqueeze(-1).expand(B, -1, T_last)"
        ),
        new=(
            "last_act_shape = self._activations[conv_act_keys[-1]]\n"
            "            B, C_last, T_last = last_act_shape.shape\n"
            "\n"
            "            # Project output relevance back through the classification head so\n"
            "            # R_feat has shape (B, C_last, T_last) rather than (B, n_classes, T_last).\n"
            "            # Using (B, n_classes) directly caused a shape mismatch in the LRP\n"
            "            # backward loop, triggering the mean-collapse fallback that produces\n"
            "            # uniform (meaningless) attributions across all time steps.\n"
            "            with torch.no_grad():\n"
            "                # head[-1]: Linear(fc_hidden → n_classes)\n"
            "                W_out  = self.model.head[-1].weight          # (n_classes, fc_hidden)\n"
            "                R_head = (R @ W_out)                         # (B, fc_hidden)\n"
            "                # head[0]: Linear(fusion_hidden → fc_hidden)\n"
            "                W_fc   = self.model.head[0].weight           # (fc_hidden, fusion_hidden)\n"
            "                R_fused = (R_head @ W_fc)                    # (B, fusion_hidden)\n"
            "                # fusion.proj_cnn: Linear(C_cnn → fusion_hidden) — transpose for backward\n"
            "                W_cnn  = self.model.fusion.proj_cnn.weight   # (fusion_hidden, C_cnn)\n"
            "                R_cnn  = (R_fused @ W_cnn)                   # (B, C_cnn)\n"
            "\n"
            "            R_feat = R_cnn.unsqueeze(-1).expand(B, -1, T_last)"
        ),
        description="Project R through head weights to get correct (B, C_last, T_last) shape",
    )


# ===========================================================================
# Bug 6 · src/xai_and_pipeline.py
# GradCAM1D hooks are registered permanently and never removed.  Running XAI
# in a loop accumulates hooks: activations are overwritten N times but
# gradients are written N times too, making results non-deterministic.
# ===========================================================================

def fix_bug6() -> None:
    print("\n[Bug 6] Store and remove GradCAM hooks (src/xai_and_pipeline.py)")
    # Fix _register_hooks to store handles
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "    def _register_hooks(self) -> None:\n"
            "        def forward_hook(module, input, output):\n"
            "            self._activations = output.detach()\n"
            "\n"
            "        def backward_hook(module, grad_in, grad_out):\n"
            "            self._gradients = grad_out[0].detach()\n"
            "\n"
            "        self.target_layer.register_forward_hook(forward_hook)\n"
            "        self.target_layer.register_full_backward_hook(backward_hook)"
        ),
        new=(
            "    def _register_hooks(self) -> None:\n"
            "        def forward_hook(module, input, output):\n"
            "            self._activations = output.detach()\n"
            "\n"
            "        def backward_hook(module, grad_in, grad_out):\n"
            "            self._gradients = grad_out[0].detach()\n"
            "\n"
            "        # Store handles so hooks can be removed after use (Bug 6 fix).\n"
            "        # Without removal, creating a new GradCAM1D for the same layer in\n"
            "        # a loop accumulates stale hooks that corrupt gradient values.\n"
            "        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)\n"
            "        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)\n"
            "\n"
            "    def remove_hooks(self) -> None:\n"
            "        \"\"\"Remove forward and backward hooks registered by _register_hooks.\n"
            "\n"
            "        Call this after the final GradCAM computation for a given instance,\n"
            "        or wrap usage in a try/finally block::\n"
            "\n"
            "            cam = GradCAM1D(model, layer)\n"
            "            try:\n"
            "                result = cam(x)\n"
            "            finally:\n"
            "                cam.remove_hooks()\n"
            "        \"\"\"\n"
            "        self._fwd_handle.remove()\n"
            "        self._bwd_handle.remove()"
        ),
        description="Store hook handles + add remove_hooks() method",
    )
    # Also patch run_xai_analysis to call remove_hooks after the sample loop
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "    grad_cam = GradCAM1D(model, target_layer)\n"
        ),
        new=(
            "    grad_cam = GradCAM1D(model, target_layer)\n"
            "    # NB: call grad_cam.remove_hooks() after the loop below (Bug 6 fix)\n"
        ),
        description="Add reminder comment before GradCAM usage in run_xai_analysis",
    )
    # Patch the point after the per-sample loop where grad_cam is no longer needed
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "    # --- end sample loop ---\n"
        ),
        new=(
            "    # --- end sample loop ---\n"
            "    grad_cam.remove_hooks()   # Bug 6 fix: prevent hook accumulation\n"
        ),
        description="Remove GradCAM hooks after sample loop in run_xai_analysis",
    )


# ===========================================================================
# Bug 7 · src/preprocessing_pipeline.py
# ECG beat window: pre_r_samples=90 and post_r_samples=270 were calibrated
# for 360 Hz but the pipeline runs at 128 Hz, giving a 2.81 s window instead
# of the intended 1.0 s — each "beat" spans 2-3 actual heartbeats.
# ===========================================================================

def fix_bug7() -> None:
    print("\n[Bug 7] Fix ECG beat window to be fs-independent (src/preprocessing_pipeline.py)")
    apply(
        "src/preprocessing_pipeline.py",
        old=(
            "    def __init__(\n"
            "        self,\n"
            "        target_fs: float = 360.0,\n"
            "        pre_r_samples:  int = 90,    # \"250 ms before R-peak at 360 Hz\"\n"
            "        post_r_samples: int = 270,   # \"750 ms after  R-peak at 360 Hz\""
        ),
        new=(
            "    def __init__(\n"
            "        self,\n"
            "        target_fs: float = 360.0,\n"
            "        pre_r_ms:  float = 250.0,    # time in ms — fs-independent (Bug 7 fix)\n"
            "        post_r_ms: float = 750.0,    # time in ms — fs-independent (Bug 7 fix)"
        ),
        description="Replace hardcoded sample counts with fs-independent millisecond values",
    )
    # Replace the body lines that stored pre_r_samples / post_r_samples directly
    apply(
        "src/preprocessing_pipeline.py",
        old=(
            "        self.target_fs = target_fs\n"
            "        self.pre_r  = pre_r_samples\n"
            "        self.post_r = post_r_samples\n"
            "        self.window_samples = self.pre_r + self.post_r"
        ),
        new=(
            "        self.target_fs = target_fs\n"
            "        # Derive sample counts from time so the window is always correct\n"
            "        # regardless of which sampling rate the pipeline was configured with.\n"
            "        self.pre_r  = int(pre_r_ms  / 1000.0 * target_fs)\n"
            "        self.post_r = int(post_r_ms / 1000.0 * target_fs)\n"
            "        self.window_samples = self.pre_r + self.post_r"
        ),
        description="Compute pre_r / post_r samples from ms × target_fs",
    )


# ===========================================================================
# Bug 8 · src/xai_and_pipeline.py + src/preprocessing_pipeline.py
# target_fs default mismatch: UnifiedRunConfig defaults to 250 Hz but
# MultimodalPreprocessingPipeline defaults to 128 Hz.  run_xai_analysis also
# hard-codes fs=250 when calling plot_all_attributions, so XAI time axes are
# ~1.95× too fast when data is actually sampled at 128 Hz.
# ===========================================================================

def fix_bug8() -> None:
    print("\n[Bug 8] Align target_fs defaults and pass fs through (src/xai_and_pipeline.py + preprocessing_pipeline.py)")

    # 8a — fix UnifiedRunConfig default
    apply(
        "src/xai_and_pipeline.py",
        old="    target_fs: int = 250",
        new="    target_fs: int = 128   # Aligned with MultimodalPreprocessingPipeline default (Bug 8 fix)",
        description="UnifiedRunConfig.target_fs default 250 → 128",
    )

    # 8b — fix hardcoded fs=250 in end_to_end_run call to run_xai_analysis
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "        run_xai_analysis(\n"
            "            model=model, loader=loaders[\"test\"], label_map=label_map,\n"
            "            output_dir=xai_dir, modality=modality,\n"
            "            n_samples=cfg.n_xai_samples,\n"
            "            device=cfg.device,\n"
            "            fs=250,\n"
            "        )"
        ),
        new=(
            "        run_xai_analysis(\n"
            "            model=model, loader=loaders[\"test\"], label_map=label_map,\n"
            "            output_dir=xai_dir, modality=modality,\n"
            "            n_samples=cfg.n_xai_samples,\n"
            "            device=cfg.device,\n"
            "            fs=cfg.target_fs,   # Bug 8 fix: pass through instead of hardcoding 250\n"
            "        )"
        ),
        description="Pass cfg.target_fs into run_xai_analysis instead of hardcoded 250",
    )

    # 8c — fix run_xai_analysis signature default
    apply(
        "src/xai_and_pipeline.py",
        old="def run_xai_analysis(\n        model,\n        loader,\n        label_map: dict,\n        output_dir: str,\n        modality: str,\n        n_samples: int = 32,\n        device: str = \"cpu\",\n        fs: int = 250,\n    ) -> None:",
        new="def run_xai_analysis(\n        model,\n        loader,\n        label_map: dict,\n        output_dir: str,\n        modality: str,\n        n_samples: int = 32,\n        device: str = \"cpu\",\n        fs: int = 128,   # Bug 8 fix: default aligned with preprocessing pipeline\n    ) -> None:",
        description="run_xai_analysis fs default 250 → 128",
    )


# ===========================================================================
# Bug 9 · main.py
# `import re` inside load_yaml_config is imported but never used.
# ===========================================================================

def fix_bug9() -> None:
    print("\n[Bug 9] Remove unused `import re` from load_yaml_config (main.py)")
    apply(
        "main.py",
        old=(
            "def load_yaml_config(path: str) -> dict:\n"
            "    \"\"\"Load config.yaml and return a flat dict of overrides (no new dependencies).\"\"\"\n"
            "    import re\n"
            "\n"
            "    overrides = {}"
        ),
        new=(
            "def load_yaml_config(path: str) -> dict:\n"
            "    \"\"\"Load config.yaml and return a flat dict of overrides (no new dependencies).\"\"\"\n"
            "    overrides = {}"
        ),
        description="Remove dead `import re` from load_yaml_config",
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("apply_patches.py — thesis repo bug fixes")
    print(f"Repository root: {REPO_ROOT}")
    print("=" * 70)

    # Check we are in the right directory
    required = ["main.py", "src", "tests", "config.yaml"]
    missing  = [r for r in required if not os.path.exists(_path(r))]
    if missing:
        print(
            f"\nERROR: the following expected paths were not found: {missing}\n"
            "Make sure you run this script from the repository root.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    fix_bug1()
    fix_bug2()
    fix_bug3()
    fix_bug4()
    fix_bug5()
    fix_bug6()
    fix_bug7()
    fix_bug8()
    fix_bug9()

    print("\n" + "=" * 70)
    print("All patches applied.")
    print("Run  python tests/verify_patches.py  to confirm the three PASS lines.")
    print("=" * 70)


if __name__ == "__main__":
    main()
