"""结果导出工具。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export_results(output_dir: str | Path, time: np.ndarray, acceleration: np.ndarray,
                   periods: np.ndarray, target_sa: np.ndarray, generated_sa: np.ndarray,
                   settings: dict) -> Path:
    """导出时程、反应谱、两张 PNG 和设置 JSON。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    error = np.abs(generated_sa - target_sa) / np.maximum(target_sa, np.finfo(float).eps)
    pd.DataFrame({"time": time, "acceleration_g": acceleration}).to_csv(out / "ground_motion.csv", index=False)
    pd.DataFrame({"period": periods, "target_Sa_g": target_sa,
                  "generated_Sa_g": generated_sa, "relative_error": error}).to_csv(
                      out / "response_spectrum.csv", index=False)
    (out / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time, acceleration, linewidth=0.8)
    ax.set(xlabel="Time (s)", ylabel="Acceleration (g)", title="Artificial Ground Motion")
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(out / "time_history.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(periods, target_sa, label="Target Spectrum", linewidth=2)
    ax.plot(periods, generated_sa, label="Generated Spectrum", linewidth=1.5)
    ax.set(xlabel="Period T (s)", ylabel="Sa (g)", title="Response Spectrum Comparison")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "response_spectrum.png", dpi=160); plt.close(fig)
    return out
