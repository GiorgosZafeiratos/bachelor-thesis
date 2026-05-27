import sys, os as _os
_src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, _os.path.abspath(_src))

"""
Run:  python verify_patches.py
Exits 0 if every check passes; prints FAIL + reason and exits 1 otherwise.
"""
import sys
import numpy as np
import torch
import torch.nn as nn


def _fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


# Verification 1 — _apply_normalisation "none" is a true no-op
def verify_normalization_none():
    """
    Calling _apply_normalisation with norm_mode="none" on an array whose
    mean is 5.0 must return an array whose mean is still 5.0 within 1e-6.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from preprocessing_pipeline import MultimodalPreprocessingPipeline

    pipeline = MultimodalPreprocessingPipeline(
        ecg_dir=None, eeg_dir=None, ppg_dir=None,
        output_dir="/tmp", norm_mode="none",
    )

    rng = np.random.default_rng(0)
    X   = (rng.standard_normal((10, 2, 250)) + 5.0).astype(np.float64)

    # FIX: was _apply_normalization (American spelling) — the actual method is
    # _apply_normalisation (British spelling), consistent with the rest of the
    # codebase. The previous spelling caused AttributeError at runtime.
    out = pipeline._apply_normalisation(X)

    assert out.dtype == np.float32, f"dtype should be float32, got {out.dtype}"
    mean_in  = float(X.mean())
    mean_out = float(out.mean())
    if abs(mean_in - mean_out) > 1e-6:
        _fail(
            f"normalization_none: mean changed from {mean_in:.6f} "
            f"to {mean_out:.6f} (delta={abs(mean_in - mean_out):.2e})"
        )
    print(f"PASS: normalization_none — mean preserved ({mean_out:.6f})")


# Verification 2 — LateFusionEnsemble fusion_rule="attention" does not crash
#                  and returns correct shape for batch_size=2

def verify_late_fusion_attention():
    from multimodal_fusion import LateFusionEnsemble

    N_CLASSES = 5
    B         = 2

    # Three minimal encoders with intentionally DIFFERENT input channel counts
    # (mimicking ECG=2, EEG=64, PPG=1) — the original bug triggered here
    class _TinyEncoder(nn.Module):
        def __init__(self, in_ch: int, n_cls: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(in_ch, n_cls),
            )
        def forward(self, x):
            return self.net(x)

    modality_models = {
        "ecg": _TinyEncoder(2,  N_CLASSES),
        "eeg": _TinyEncoder(64, N_CLASSES),
        "ppg": _TinyEncoder(1,  N_CLASSES),
    }

    ensemble = LateFusionEnsemble(
        modality_models=modality_models,
        n_classes=N_CLASSES,
        fusion_rule="attention",
    )
    ensemble.eval()

    inputs = {
        "ecg": torch.randn(B,  2, 250),
        "eeg": torch.randn(B, 64, 250),
        "ppg": torch.randn(B,  1, 250),
    }

    try:
        with torch.no_grad():
            out = ensemble(inputs)
    except RuntimeError as e:
        _fail(f"late_fusion_attention raised RuntimeError: {e}")

    expected = (B, N_CLASSES)
    if tuple(out.shape) != expected:
        _fail(
            f"late_fusion_attention: output shape {tuple(out.shape)} "
            f"!= expected {expected}"
        )
    print(f"PASS: late_fusion_attention — output shape {tuple(out.shape)}")


# Verification 3 — config.yaml loads all 12 required keys via load_yaml_config
def verify_config_yaml():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from main import load_yaml_config

    # FIX: was os.path.join(os.path.dirname(__file__), "config.yaml") which
    # resolves to tests/config.yaml — a path that does not exist. config.yaml
    # lives at the repository root, one directory above tests/.
    yaml_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    )
    if not os.path.exists(yaml_path):
        _fail(f"config.yaml not found at expected path: {yaml_path}")

    cfg = load_yaml_config(yaml_path)

    required_keys = [
        "raw_data_dir", "processed_dir", "output_dir",
        "modalities", "norm_strategy",
        "epochs", "batch_size", "lr", "patience", "seed",
        "num_workers", "n_xai_samples",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        _fail(f"config.yaml missing keys: {missing}")

    if not isinstance(cfg["modalities"], list):
        _fail(f"config.yaml 'modalities' should be a list, got {type(cfg['modalities'])}")

    if not isinstance(cfg["epochs"], int):
        _fail(f"config.yaml 'epochs' should be int, got {type(cfg['epochs'])}")

    print(f"PASS: config_yaml — {len(cfg)} keys loaded, all 12 required keys present")


if __name__ == "__main__":
    verify_normalization_none()
    verify_late_fusion_attention()
    verify_config_yaml()
    print("\nAll verifications passed.")
    
