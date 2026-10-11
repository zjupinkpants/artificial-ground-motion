"""Newmark-beta 反应谱计算、误差评价和频域迭代谱匹配。"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ground_motion import baseline_correct, envelope_function


@dataclass
class ErrorMetrics:
    mean_error: float
    max_error: float
    within_10_percent: float
    within_20_percent: float


def compute_response_spectrum(
    acceleration: np.ndarray, dt: float, periods: np.ndarray, damping: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用平均加速度 Newmark-beta 法计算 Sd、Sv 和绝对加速度谱 Sa。

    输入地面加速度 ag 单位 g；内部乘 g0 转为 m/s²。相对响应满足
    u¨ + 2ζωu˙ + ω²u = -ag，绝对加速度为 u¨+ag = -2ζωu˙-ω²u。
    Sd、Sv 分别为 m、m/s，Sa 输出仍为 g。
    """
    a_g = np.asarray(acceleration, dtype=float)
    t = np.asarray(periods, dtype=float)
    if dt <= 0 or a_g.ndim != 1 or len(a_g) < 2:
        raise ValueError("dt 必须大于 0，且加速度必须为有效一维数组")
    if np.any(t <= 0) or not 0 <= damping < 1:
        raise ValueError("周期必须大于 0，阻尼比应在 [0,1) 内")
    if not np.all(np.isfinite(a_g)):
        raise ValueError("加速度中存在 NaN 或 Inf")
    g0 = 9.80665
    ag = a_g * g0
    omega = 2.0 * np.pi / t
    stiffness = omega**2
    damping_c = 2.0 * damping * omega
    beta, gamma = 0.25, 0.5
    c0 = 1.0 / (beta * dt**2)
    c1 = gamma / (beta * dt)
    c2 = 1.0 / (beta * dt)
    c3 = 1.0 / (2.0 * beta) - 1.0
    c4 = gamma / beta - 1.0
    c5 = dt * (gamma / (2.0 * beta) - 1.0)
    effective_k = stiffness + c0 + c1 * damping_c
    u = np.zeros_like(t)
    v = np.zeros_like(t)
    rel_acc = np.full_like(t, -ag[0])
    max_u = np.zeros_like(t)
    max_v = np.zeros_like(t)
    max_abs_acc = np.abs(-damping_c * v - stiffness * u)
    for i in range(len(ag) - 1):
        effective_p = (-ag[i + 1] + c0 * u + c2 * v + c3 * rel_acc
                       + damping_c * (c1 * u + c4 * v + c5 * rel_acc))
        u_new = effective_p / effective_k
        rel_acc_new = c0 * (u_new - u) - c2 * v - c3 * rel_acc
        v_new = v + dt * ((1.0 - gamma) * rel_acc + gamma * rel_acc_new)
        u, v, rel_acc = u_new, v_new, rel_acc_new
        max_u = np.maximum(max_u, np.abs(u))
        max_v = np.maximum(max_v, np.abs(v))
        max_abs_acc = np.maximum(max_abs_acc, np.abs(-damping_c * v - stiffness * u))
    return max_u, max_v, max_abs_acc / g0


def evaluate_error(current_sa: np.ndarray, target_sa: np.ndarray, periods: np.ndarray,
                   match_range: tuple[float, float] = (0.1, 4.0)) -> tuple[np.ndarray, ErrorMetrics]:
    """计算逐点相对误差及重点周期区间内的汇总指标。"""
    current, target, periods = map(lambda x: np.asarray(x, dtype=float), (current_sa, target_sa, periods))
    if np.any(target <= 0):
        raise ValueError("目标谱 Sa 必须全部大于 0")
    error = np.abs(current - target) / np.maximum(target, np.finfo(float).eps)
    mask = (periods >= match_range[0]) & (periods <= match_range[1])
    if not np.any(mask):
        raise ValueError("重点匹配区间内没有周期点")
    selected = error[mask]
    metrics = ErrorMetrics(float(np.mean(selected)), float(np.max(selected)),
                           float(np.mean(selected <= 0.10)), float(np.mean(selected <= 0.20)))
    return error, metrics


def match_spectrum(acceleration: np.ndarray, dt: float, periods: np.ndarray,
                   target_sa: np.ndarray, damping: float = 0.05, max_iterations: int = 20,
                   tolerance: float = 0.10, relaxation_alpha: float = 0.5,
                   match_range: tuple[float, float] = (0.1, 4.0),
                   callback: Callable[[int, ErrorMetrics], None] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, list[ErrorMetrics]]:
    """通过平滑频域幅值修正迭代匹配目标反应谱。"""
    if max_iterations < 1 or tolerance <= 0 or not 0 < relaxation_alpha <= 1:
        raise ValueError("迭代数、容差和松弛系数设置无效")
    a = np.asarray(acceleration, dtype=float).copy()
    n = len(a)
    freq = np.fft.rfftfreq(n, dt)
    history: list[ErrorMetrics] = []
    duration = (n - 1) * dt
    env = envelope_function(np.arange(n) * dt, duration)
    for iteration in range(1, max_iterations + 1):
        _, _, current_sa = compute_response_spectrum(a, dt, periods, damping)
        _, metrics = evaluate_error(current_sa, target_sa, periods, match_range)
        history.append(metrics)
        if callback:
            callback(iteration, metrics)
        if metrics.mean_error <= tolerance:
            break
        ratio = np.clip(target_sa / np.maximum(current_sa, 1e-8), 0.5, 2.0)
        ratio = gaussian_filter1d(ratio, sigma=2.0, mode="nearest")
        # 周期升序对应频率降序，插值前需将映射频率排序。
        mapped_f = 1.0 / periods
        correction = np.interp(freq, mapped_f[::-1], ratio[::-1], left=1.0, right=1.0)
        correction = gaussian_filter1d(correction, sigma=2.0, mode="nearest") ** relaxation_alpha
        correction[0] = 0.0
        fourier = np.fft.rfft(a)
        fourier *= correction
        fourier[0] = 0.0
        a = np.fft.irfft(fourier, n=n)
        # 温和重施包络，避免每轮重复包络导致端部过度衰减。
        a = baseline_correct(a * (0.9 + 0.1 * env))
    _, _, final_sa = compute_response_spectrum(a, dt, periods, damping)
    return a, final_sa, history
