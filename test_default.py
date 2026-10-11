"""默认参数端到端验证脚本：python test_default.py"""

from pathlib import Path

import numpy as np

from ground_motion import generate_initial_motion
from matching import evaluate_error, match_spectrum
from spectrum import SpectrumParameters, generic_target_spectrum, make_periods
from utils import export_results


def main() -> None:
    periods = make_periods(0.02, 6.0, 0.02)
    target = generic_target_spectrum(periods, SpectrumParameters(smax=0.20, tg=0.40))
    time, initial = generate_initial_motion(periods, target, 30.0, 0.01, 42)
    acceleration, generated, history = match_spectrum(initial, 0.01, periods, target, 0.05, 20, 0.10, 0.5)
    error, metrics = evaluate_error(generated, target, periods, (0.1, 4.0))
    assert len(time) == 3001 and len(acceleration) == len(time)
    assert len(np.fft.irfft(np.fft.rfft(acceleration), n=len(acceleration))) == len(acceleration)
    assert np.all(np.isfinite(acceleration)) and np.all(np.isfinite(generated)) and np.all(np.isfinite(error))
    settings = {"duration": 30.0, "dt": 0.01, "Tmin": 0.02, "Tmax": 6.0, "dT": 0.02,
                "damping": 0.05, "Smax": 0.20, "Tg": 0.40, "max_iterations": 20,
                "tolerance": 0.10, "relaxation_alpha": 0.5, "random_seed": 42,
                "note": "PGA scaling is intentionally not applied during spectrum matching."}
    out = export_results(Path(__file__).parent / "outputs" / "default_case", time, acceleration, periods, target, generated, settings)
    print(f"PGA: {np.max(np.abs(acceleration)):.6f} g")
    print(f"Mean spectral error (0.1-4.0 s): {metrics.mean_error:.2%}")
    print(f"Max spectral error (0.1-4.0 s): {metrics.max_error:.2%}")
    print(f"Points within 10%: {metrics.within_10_percent:.2%}")
    print(f"Points within 20%: {metrics.within_20_percent:.2%}")
    print(f"Iterations: {len(history)}")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
