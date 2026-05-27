import os
import shutil
import sys
import textwrap

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_failures = []


def _path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def read(rel):
    with open(_path(rel), encoding="utf-8") as fh:
        return fh.read()


def write(rel, content):
    dest = _path(rel)
    bak = dest + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(dest, bak)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(content)


def apply(rel, old, new, description):
    src = read(rel)
    if old not in src:
        if new in src:
            print(f"  SKIP  {description!r}  (already applied in {rel})")
        else:
            msg = f"FAIL  {description!r}  — anchor not found in {rel}"
            print(f"  {msg}", file=sys.stderr)
            print(f"        Looking for:\n{textwrap.indent(repr(old[:120]), '        ')}", file=sys.stderr)
            _failures.append(msg)
        return
    write(rel, src.replace(old, new, 1))
    print(f"  PASS  {description!r}  → {rel}")


# ---------------------------------------------------------------------------
# Original 9 bugs (unchanged from v1)
# ---------------------------------------------------------------------------

def fix_bug1():
    print("\n[Bug 1] Fix _apply_normalization → _apply_normalisation")
    apply(
        "tests/verify_patches.py",
        old="out = pipeline._apply_normalization(X)",
        new="out = pipeline._apply_normalisation(X)",
        description="British-spelling fix",
    )


def fix_bug2():
    print("\n[Bug 2] Fix config.yaml path in verify_patches.py")
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
        description="Resolve config.yaml one level up from tests/",
    )


def fix_bug3():
    print("\n[Bug 3] Fix YAML always overriding CLI flags")
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
            "        explicit_cli_keys = {\n"
            "            a.lstrip('-').replace('-', '_').split('=')[0]\n"
            "            for a in sys.argv[1:]\n"
            "            if a.startswith('--')\n"
            "        }\n"
            "        for key, val in overrides.items():\n"
            "            if key not in explicit_cli_keys:   # CLI wins; YAML fills the rest\n"
            "                setattr(args, key, val)"
        ),
        description="CLI keys beat YAML",
    )


def fix_bug4():
    print("\n[Bug 4] Guard SEBlock against se_ratio=0 in ResidualConvBlock")
    apply(
        "src/hybrid_model.py",
        old="self.se = SEBlock(out_channels, se_ratio)",
        new="self.se = SEBlock(out_channels, se_ratio) if se_ratio > 0 else nn.Identity()",
        description="se_ratio=0 → nn.Identity()",
    )


def fix_bug5():
    print("\n[Bug 5] Fix R_feat shape in LRPExplainer (n_classes → C_last)")
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
            "            # Project output relevance back through the head so R_feat has\n"
            "            # shape (B, C_last, T_last) — not (B, n_classes, T_last).\n"
            "            with torch.no_grad():\n"
            "                W_out   = self.model.head[-1].weight          # (n_classes, fc_hidden)\n"
            "                R_head  = (R @ W_out)                         # (B, fc_hidden)\n"
            "                W_fc    = self.model.head[0].weight           # (fc_hidden, fusion_hidden)\n"
            "                R_fused = (R_head @ W_fc)                     # (B, fusion_hidden)\n"
            "                W_cnn   = self.model.fusion.proj_cnn.weight   # (fusion_hidden, C_cnn)\n"
            "                R_cnn   = (R_fused @ W_cnn)                   # (B, C_cnn)\n"
            "            R_feat = R_cnn.unsqueeze(-1).expand(B, -1, T_last)"
        ),
        description="Project R through head weights to (B, C_last, T_last)",
    )


def fix_bug6():
    print("\n[Bug 6] Store GradCAM hook handles + remove_hooks() + try/finally in caller")

    # Part A: store handles and add remove_hooks()
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
            "        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)\n"
            "        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)\n"
            "\n"
            "    def remove_hooks(self) -> None:\n"
            "        \"\"\"Remove registered hooks. Call after final use to avoid accumulation.\"\"\"\n"
            "        self._fwd_handle.remove()\n"
            "        self._bwd_handle.remove()"
        ),
        description="Store hook handles + add remove_hooks()",
    )

    # Part B: wrap GradCAM usage in run_xai_analysis with try/finally
    # Use a real anchor: the pattern of creating GradCAM and iterating the loader.
    # This replaces the brittle "# --- end sample loop ---" invented comment approach.
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "    grad_cam = GradCAM1D(model, target_layer)\n"
            "    for i, (x, y) in enumerate(loader):"
        ),
        new=(
            "    grad_cam = GradCAM1D(model, target_layer)\n"
            "    try:\n"
            "      for i, (x, y) in enumerate(loader):"
        ),
        description="Open try block around GradCAM sample loop",
    )
    # Close the try with finally: remove_hooks.
    # Anchor: the block that writes/saves XAI figures and breaks after n_samples.
    # We use the n_samples break since it's the only guaranteed exit of the loop.
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "        if i >= n_samples - 1:\n"
            "            break\n"
            "    visualizer"
        ),
        new=(
            "        if i >= n_samples - 1:\n"
            "            break\n"
            "    finally:\n"
            "        grad_cam.remove_hooks()   # Bug 6: prevent hook accumulation\n"
            "    visualizer"
        ),
        description="Close try/finally with grad_cam.remove_hooks()",
    )


def fix_bug7():
    print("\n[Bug 7] Fix ECG beat window to be fs-independent")
    apply(
        "src/preprocessing_pipeline.py",
        old=(
            "    def __init__(\n"
            "        self,\n"
            "        target_fs: float = 360.0,\n"
            '        pre_r_samples:  int = 90,    # "250 ms before R-peak at 360 Hz"\n'
            '        post_r_samples: int = 270,   # "750 ms after  R-peak at 360 Hz"'
        ),
        new=(
            "    def __init__(\n"
            "        self,\n"
            "        target_fs: float = 360.0,\n"
            "        pre_r_ms:  float = 250.0,    # time in ms — fs-independent\n"
            "        post_r_ms: float = 750.0,    # time in ms — fs-independent"
        ),
        description="Replace hardcoded sample counts with ms values",
    )
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
            "        self.pre_r  = int(pre_r_ms  / 1000.0 * target_fs)\n"
            "        self.post_r = int(post_r_ms / 1000.0 * target_fs)\n"
            "        self.window_samples = self.pre_r + self.post_r"
        ),
        description="Compute samples from ms × target_fs",
    )


def fix_bug8():
    print("\n[Bug 8] Align target_fs defaults and stop hardcoding fs=250")
    apply(
        "src/xai_and_pipeline.py",
        old="    target_fs: int = 250",
        new="    target_fs: int = 128   # aligned with MultimodalPreprocessingPipeline default",
        description="UnifiedRunConfig.target_fs 250 → 128",
    )
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
            "            fs=cfg.target_fs,\n"
            "        )"
        ),
        description="Pass cfg.target_fs instead of hardcoded 250",
    )
    apply(
        "src/xai_and_pipeline.py",
        old="    fs: int = 250,",
        new="    fs: int = 128,",
        description="run_xai_analysis default fs 250 → 128",
    )


def fix_bug9():
    print("\n[Bug 9] Remove unused `import re` from load_yaml_config")
    apply(
        "main.py",
        old=(
            "def load_yaml_config(path: str) -> dict:\n"
            '    """Load config.yaml and return a flat dict of overrides (no new dependencies)."""\n'
            "    import re\n"
            "\n"
            "    overrides = {}"
        ),
        new=(
            "def load_yaml_config(path: str) -> dict:\n"
            '    """Load config.yaml and return a flat dict of overrides (no new dependencies)."""\n'
            "    overrides = {}"
        ),
        description="Remove dead `import re`",
    )


# ---------------------------------------------------------------------------
# NEW bugs found in re-review
# ---------------------------------------------------------------------------

def fix_new1():
    """LearnableNorm: divide by (σ + eps) to prevent NaN on flat windows."""
    print("\n[NEW-1] LearnableNorm: add eps guard on σ to prevent NaN/Inf")
    # The exact implementation varies; patch the most likely formulation.
    # Pattern 1: inline division
    apply(
        "src/hybrid_model.py",
        old="x_norm = (x - mu) / sigma",
        new="x_norm = (x - mu) / (sigma + 1e-8)  # eps guard: prevents NaN on flat windows",
        description="LearnableNorm: (x-μ)/(σ+eps)",
    )
    # Pattern 2: std() used directly
    apply(
        "src/hybrid_model.py",
        old="x_norm = (x - mu) / std",
        new="x_norm = (x - mu) / (std + 1e-8)    # eps guard: prevents NaN on flat windows",
        description="LearnableNorm: (x-μ)/(std+eps)",
    )


def fix_new2():
    """GradCAM: ensure remove_hooks() is called via try/finally in run_xai_analysis."""
    # Handled as part of the reworked fix_bug6() above (try/finally approach).
    print("\n[NEW-2] GradCAM try/finally — covered by revised fix_bug6()")


def fix_new5():
    """IntegratedGradients1D: ensure model is in eval mode during attribution."""
    print("\n[NEW-5] IntegratedGradients1D: ensure model.eval() before IG attribution")
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "    def attribute(\n"
            "        self,\n"
            "        x: torch.Tensor,\n"
            "        target_class: Optional[int] = None,\n"
            "        n_steps: int = 50,\n"
            "    ) -> torch.Tensor:"
        ),
        new=(
            "    def attribute(\n"
            "        self,\n"
            "        x: torch.Tensor,\n"
            "        target_class: Optional[int] = None,\n"
            "        n_steps: int = 50,\n"
            "    ) -> torch.Tensor:\n"
            "        # Ensure eval mode: BatchNorm must use running stats, not batch stats,\n"
            "        # so IG attributions are reproducible regardless of batch composition.\n"
            "        _was_training = self.model.training\n"
            "        self.model.eval()"
        ),
        description="IG attribute(): snapshot and force eval mode",
    )
    # Restore training mode at the end of the method — anchor on the return statement
    apply(
        "src/xai_and_pipeline.py",
        old=(
            "        integrated_grads = ig_sum / n_steps\n"
            "        return integrated_grads"
        ),
        new=(
            "        integrated_grads = ig_sum / n_steps\n"
            "        if _was_training:\n"
            "            self.model.train()   # restore original mode\n"
            "        return integrated_grads"
        ),
        description="IG attribute(): restore training mode on return",
    )


def fix_new3_config_yaml():
    """Add target_fs to config.yaml so it's overridable without editing source."""
    print("\n[NEW-3] Add target_fs to config.yaml")
    apply(
        "config.yaml",
        old="num_workers: 4",
        new="num_workers: 4\ntarget_fs: 128      # sampling rate for XAI time axes and ECG segmentation",
        description="Add target_fs: 128 to config.yaml",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("apply_patches_v2.py — thesis repo bug fixes (updated)")
    print(f"Repository root: {REPO_ROOT}")
    print("=" * 70)

    required = ["main.py", "src", "tests", "config.yaml"]
    missing = [r for r in required if not os.path.exists(_path(r))]
    if missing:
        print(f"\nERROR: expected paths not found: {missing}\nRun from the repo root.\n",
              file=sys.stderr)
        sys.exit(1)

    # Original 9 bugs
    fix_bug1()
    fix_bug2()
    fix_bug3()
    fix_bug4()
    fix_bug5()
    fix_bug6()
    fix_bug7()
    fix_bug8()
    fix_bug9()

    # New findings from re-review
    fix_new1()
    fix_new2()
    fix_new5()
    fix_new3_config_yaml()

    print("\n" + "=" * 70)
    if _failures:
        print(f"COMPLETED WITH {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"  ✗  {f}")
        print("\nFor each FAIL, manually apply the patch described in PATCHES.md.")
        sys.exit(1)
    else:
        print("All patches applied successfully.")
        print("Run  python tests/verify_patches.py  to confirm 3× PASS.")
    print("=" * 70)


if __name__ == "__main__":
    main()
