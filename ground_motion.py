"""初始人工地震动、包络和基线校正。加速度单位为 g。"""

import numpy as np
from scipy import signal


def envelope_function(time: np.ndarray, duration: float) -> np.ndarray:
    """生成上升—平稳—指数衰减三段包络。"""
    if duration <= 0:
        raise ValueError("duration 必须大于 0")
    t = np.asarray(time, dtype=float)
    t1, t2 = 0.15 * duration, 0.55 * duration
    env = np.ones_like(t)
    env[t <= t1] = (t[t <= t1] / max(t1, np.finfo(float).eps)) ** 2
    tail = t > t2
    env[tail] = np.exp(-3.5 * (t[tail] - t2) / max(duration - t2, np.finfo(float).eps))
    return env


def baseline_correct(acceleration: np.ndarray) -> np.ndarray:
    """去均值并去除线性趋势，不默认进行高通滤波。"""
    a = np.asarray(acceleration, dtype=float)
    if a.ndim != 1 or len(a) < 3:
        raise ValueError("加速度时程必须是一维且至少包含 3 个点")
    return signal.detrend(a - np.mean(a), type="linear")


def generate_initial_motion(
    periods: np.ndarray,
    target_sa: np.ndarray,
    duration: float = 30.0,
    dt: float = 0.01,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """用目标谱塑造傅里叶幅值、随机相位和 IFFT 生成初始时程。"""
    if dt <= 0 or duration <= 0:
        raise ValueError("dt 和 duration 必须大于 0")
    n = int(round(duration / dt)) + 1
    time = np.arange(n, dtype=float) * dt
    frequencies = np.fft.rfftfreq(n, dt)
    # 将目标 Sa(1/f) 插值为初始幅值形状；超出目标周期范围处平滑压低。
    positive_f = np.maximum(frequencies, 1.0 / (10.0 * np.max(periods)))
    mapped_t = 1.0 / positive_f
    amplitude = np.interp(mapped_t, periods, target_sa, left=target_sa[0], right=target_sa[-1])
    amplitude *= frequencies / np.sqrt(frequencies**2 + 0.08**2)  # 抑制零频漂移
    amplitude *= np.exp(-(frequencies / (0.45 / dt)) ** 8)  # Nyquist 附近渐降
    amplitude[0] = 0.0
    rng = np.random.default_rng(random_seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0
    if n % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    acceleration = np.fft.irfft(spectrum, n=n)
    acceleration = baseline_correct(acceleration * envelope_function(time, duration))
    # 初始波缩放至目标谱量级，之后由谱匹配负责精细调整。
    peak = np.max(np.abs(acceleration))
    if peak <= np.finfo(float).eps:
        raise RuntimeError("初始人工波幅值为零")
    acceleration *= float(np.max(target_sa)) / peak
    return time, acceleration


def scale_to_pga(acceleration: np.ndarray, target_pga: float) -> tuple[np.ndarray, float]:
    """将整条时程统一缩放到指定 PGA（g），并返回缩放系数。"""
    if target_pga <= 0:
        raise ValueError("目标 PGA 必须大于 0")
    current = float(np.max(np.abs(acceleration)))
    if current <= np.finfo(float).eps:
        raise ValueError("当前时程 PGA 为零，无法缩放")
    factor = target_pga / current
    return np.asarray(acceleration) * factor, factor
