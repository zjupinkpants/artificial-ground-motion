"""目标反应谱的生成与文件读取。加速度单位统一采用 g。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectrumParameters:
    """通用分段目标谱参数，便于将来替换为不同国家规范。"""

    smax: float = 0.20
    tg: float = 0.40
    rise_start_ratio: float = 0.45
    rise_end_fraction: float = 0.25
    medium_exponent: float = 0.9
    long_exponent: float = 2.0
    long_period_factor: float = 5.0


def make_periods(tmin: float, tmax: float, step: float) -> np.ndarray:
    """生成包含端点附近的等间隔周期数组。"""
    if tmin <= 0 or tmax <= 0 or step <= 0 or tmin >= tmax:
        raise ValueError("周期参数必须满足 0 < Tmin < Tmax 且 dT > 0")
    return np.arange(tmin, tmax + 0.5 * step, step, dtype=float)


def generic_target_spectrum(periods: np.ndarray, params: SpectrumParameters) -> np.ndarray:
    """计算通用四段式目标加速度反应谱，输出单位 g。

    0~0.25Tg 线性上升；0.25Tg~Tg 为平台；Tg~5Tg 按 T^-p
    衰减；更长周期以连续方式按 T^-q 继续衰减。
    """
    t = np.asarray(periods, dtype=float)
    if np.any(t < 0) or params.smax <= 0 or params.tg <= 0:
        raise ValueError("周期不可为负，Smax 和 Tg 必须大于 0")
    t1 = params.rise_end_fraction * params.tg
    t2 = params.tg
    t3 = params.long_period_factor * params.tg
    sa = np.empty_like(t)
    rise = t <= t1
    plateau = (t > t1) & (t <= t2)
    medium = (t > t2) & (t <= t3)
    long = t > t3
    # 短周期从 rise_start_ratio*Smax 线性上升至平台值。
    sa[rise] = params.smax * (params.rise_start_ratio + (1.0 - params.rise_start_ratio) * t[rise] / t1)
    sa[plateau] = params.smax
    sa[medium] = params.smax * (params.tg / t[medium]) ** params.medium_exponent
    sa_t3 = params.smax * (params.tg / t3) ** params.medium_exponent
    sa[long] = sa_t3 * (t3 / t[long]) ** params.long_exponent
    return sa


def load_target_spectrum(path: str | Path, internal_periods: np.ndarray) -> np.ndarray:
    """读取含 T、Sa 两列的 CSV/TXT，并插值到内部周期点。"""
    path = Path(path)
    try:
        frame = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception as exc:
        raise ValueError(f"无法读取目标谱文件：{exc}") from exc
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    if "t" not in normalized or "sa" not in normalized:
        raise ValueError("目标谱文件必须包含 T,Sa 两列")
    data = frame[[normalized["t"], normalized["sa"]]].apply(pd.to_numeric, errors="coerce").dropna()
    values = data.to_numpy(dtype=float)
    if len(values) < 2 or np.any(values[:, 0] <= 0) or np.any(values[:, 1] <= 0):
        raise ValueError("目标谱至少需要两个有效点，且 T、Sa 均须大于 0")
    order = np.argsort(values[:, 0])
    x, y = values[order, 0], values[order, 1]
    if np.any(np.diff(x) <= 0):
        raise ValueError("目标谱周期 T 必须唯一且严格递增")
    return np.interp(np.asarray(internal_periods), x, y)
