"""tkinter 图形界面。"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ground_motion import generate_initial_motion, scale_to_pga
from matching import compute_response_spectrum, evaluate_error, match_spectrum
from spectrum import SpectrumParameters, generic_target_spectrum, load_target_spectrum, make_periods
from utils import export_results


class GroundMotionApp(tk.Tk):
    """人工地震波生成与谱匹配主窗口。"""

    DEFAULTS = {"Smax/PGA (g)": "0.20", "Tg (s)": "0.40", "damping": "0.05",
                "Tmin (s)": "0.02", "Tmax (s)": "6.0", "dT (s)": "0.02",
                "duration (s)": "30.0", "dt (s)": "0.01", "random seed": "42",
                "max iterations": "20", "tolerance": "0.10", "alpha": "0.5",
                "match Tmin": "0.1", "match Tmax": "4.0"}

    def __init__(self) -> None:
        super().__init__()
        self.title("基于目标反应谱生成人工地震波")
        self.geometry("1320x820")
        self.vars = {k: tk.StringVar(value=v) for k, v in self.DEFAULTS.items()}
        self.periods = self.target_sa = self.time = self.acceleration = self.generated_sa = None
        self._build()

    def _build(self) -> None:
        left = ttk.Frame(self, padding=10); left.pack(side=tk.LEFT, fill=tk.Y)
        for row, (label, var) in enumerate(self.vars.items()):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(left, textvariable=var, width=14).grid(row=row, column=1, pady=2)
        row = len(self.vars)
        for text, command in (("生成目标谱", self.create_target), ("导入目标谱 CSV/TXT", self.import_target),
                              ("生成初始人工波", self.create_initial), ("开始谱匹配", self.start_matching),
                              ("按输入 PGA 缩放", self.apply_pga), ("导出结果", self.export)):
            ttk.Button(left, text=text, command=command).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3); row += 1
        self.status = tk.Text(left, width=34, height=12); self.status.grid(row=row, column=0, columnspan=2, pady=8)
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.axes = [self.figure.add_subplot(311), self.figure.add_subplot(312), self.figure.add_subplot(313)]
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _values(self) -> dict:
        try:
            p = {k: float(v.get()) for k, v in self.vars.items() if k != "random seed" and k != "max iterations"}
            p["random seed"] = int(self.vars["random seed"].get()); p["max iterations"] = int(self.vars["max iterations"].get())
        except ValueError as exc:
            raise ValueError("所有参数必须为有效数字") from exc
        if p["dt (s)"] <= 0 or p["duration (s)"] <= 0 or p["Tmin (s)"] >= p["Tmax (s)"]:
            raise ValueError("请检查 dt、duration 和周期上下限")
        return p

    def create_target(self) -> None:
        try:
            p = self._values(); self.periods = make_periods(p["Tmin (s)"], p["Tmax (s)"], p["dT (s)"])
            self.target_sa = generic_target_spectrum(self.periods, SpectrumParameters(smax=p["Smax/PGA (g)"], tg=p["Tg (s)"]))
            self._plot(); self._log("目标谱已生成。")
        except Exception as exc: messagebox.showerror("参数错误", str(exc))

    def import_target(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Spectrum", "*.csv *.txt"), ("All", "*.*")])
        if not path: return
        try:
            p = self._values(); self.periods = make_periods(p["Tmin (s)"], p["Tmax (s)"], p["dT (s)"])
            self.target_sa = load_target_spectrum(path, self.periods); self._plot(); self._log(f"已导入：{path}")
        except Exception as exc: messagebox.showerror("导入失败", str(exc))

    def create_initial(self) -> None:
        try:
            if self.target_sa is None: self.create_target()
            if self.target_sa is None: return
            p = self._values(); self.time, self.acceleration = generate_initial_motion(self.periods, self.target_sa, p["duration (s)"], p["dt (s)"], p["random seed"])
            _, _, self.generated_sa = compute_response_spectrum(self.acceleration, p["dt (s)"], self.periods, p["damping"])
            self._plot(); self._log("初始人工波已生成。")
        except Exception as exc: messagebox.showerror("生成失败", str(exc))

    def start_matching(self) -> None:
        if self.acceleration is None: self.create_initial()
        if self.acceleration is None: return
        try:
            # tkinter 变量只能在主线程读取，因此启动工作线程前取得参数快照。
            parameters = self._values()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc)); return
        self._log("开始匹配……")
        threading.Thread(target=self._matching_worker, args=(parameters,), daemon=True).start()

    def _matching_worker(self, p: dict) -> None:
        try:
            def report(i, m): self.after(0, self._log, f"Iteration {i}: Mean {m.mean_error:.2%}, Max {m.max_error:.2%}")
            a, sa, _ = match_spectrum(self.acceleration, p["dt (s)"], self.periods, self.target_sa,
                p["damping"], p["max iterations"], p["tolerance"], p["alpha"],
                (p["match Tmin"], p["match Tmax"]), report)
            self.acceleration, self.generated_sa = a, sa
            self.after(0, self._plot); self.after(0, self._log, "谱匹配完成。")
        except Exception as exc: self.after(0, messagebox.showerror, "匹配失败", str(exc))

    def apply_pga(self) -> None:
        try:
            if self.acceleration is None: raise ValueError("请先生成时程")
            p = self._values(); self.acceleration, factor = scale_to_pga(self.acceleration, p["Smax/PGA (g)"])
            _, _, self.generated_sa = compute_response_spectrum(self.acceleration, p["dt (s)"], self.periods, p["damping"])
            self._plot(); self._log(f"PGA 缩放系数：{factor:.5g}（缩放后反应谱已重算）")
        except Exception as exc: messagebox.showerror("缩放失败", str(exc))

    def export(self) -> None:
        try:
            if self.generated_sa is None: raise ValueError("没有可导出的完整结果")
            directory = filedialog.askdirectory()
            if directory:
                export_results(directory, self.time, self.acceleration, self.periods, self.target_sa, self.generated_sa, self._values())
                self._log(f"结果已导出至：{directory}")
        except Exception as exc: messagebox.showerror("导出失败", str(exc))

    def _plot(self) -> None:
        for ax in self.axes: ax.clear(); ax.grid(True, alpha=0.25)
        if self.time is not None: self.axes[0].plot(self.time, self.acceleration, linewidth=0.7)
        self.axes[0].set(xlabel="Time (s)", ylabel="Acceleration (g)")
        if self.periods is not None: self.axes[1].plot(self.periods, self.target_sa, label="Target Spectrum", linewidth=2)
        if self.generated_sa is not None: self.axes[1].plot(self.periods, self.generated_sa, label="Generated Spectrum")
        if self.periods is not None: self.axes[1].legend()
        self.axes[1].set(xlabel="Period T (s)", ylabel="Sa (g)")
        if self.acceleration is not None:
            p = self._values(); f = np.fft.rfftfreq(len(self.acceleration), p["dt (s)"]); amp = 2 * np.abs(np.fft.rfft(self.acceleration)) / len(self.acceleration)
            self.axes[2].plot(f, amp, linewidth=0.8); self.axes[2].set_xlim(0, min(25, f[-1]))
        self.axes[2].set(xlabel="Frequency (Hz)", ylabel="Fourier amplitude (g)")
        self.figure.tight_layout(); self.canvas.draw_idle()

    def _log(self, text: str) -> None:
        self.status.insert(tk.END, text + "\n"); self.status.see(tk.END)


def run_gui() -> None:
    GroundMotionApp().mainloop()
