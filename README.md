# 基于目标反应谱生成人工地震波

一个用于教学、研究和工程原型验证的 Python 桌面程序。程序输出的人工地震波不能替代经过审查、选波和一致性验证的正式设计地震动。

这是一个清晰、可扩展的第一版工程程序。加速度统一使用 **g**；Newmark 计算内部转换为 m/s²，输出的 `Sa` 再转换为 g。

## 安装与运行

```bash
python -m pip install -r requirements.txt
python main.py
```

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖。程序需要能够显示桌面窗口的 Windows/macOS/Linux 环境。

Windows 官方 Python 通常自带 tkinter；若系统没有 tkinter，请通过 Python 安装器补装 Tcl/Tk。默认参数可直接依次点击“生成目标谱”“生成初始人工波”“开始谱匹配”。匹配在后台线程运行，避免 GUI 假死。

无图形界面的端到端验证：

```bash
python test_default.py
```

结果写入 `outputs/default_case/`。

## 算法概述

- `spectrum.py`：四段式通用示例目标谱，以及 CSV/TXT 的 `T,Sa` 导入与插值。公式参数集中在 `SpectrumParameters`，后续可增加中国规范、EC8 或 ASCE 7 实现。
- `ground_motion.py`：目标谱映射为傅里叶幅值形状，配合随机相位 IFFT；使用上升、平稳、指数衰减包络；随后去均值和线性趋势。
- `matching.py`：平均加速度 Newmark-beta（β=1/4、γ=1/2）求解 SDOF 相对响应，并取绝对加速度峰值为 `Sa`。谱匹配将 `target/current` 限制在 0.5~2.0，平滑后按 `ratio ** alpha` 修正傅里叶幅值。
- `gui.py`：参数、导入、生成、匹配、独立 PGA 缩放、绘图与导出。

运动方程为 `u¨ + 2ζωu˙ + ω²u = -ag`。相对位移、速度分别是 `u`、`u˙`；绝对加速度为 `u¨ + ag = -2ζωu˙ - ω²u`。`T=0` 不直接进入振子计算；程序要求周期严格大于 0，PGA 可直接由地面时程峰值获得。

## 输入和导出

目标谱文件须有表头 `T,Sa`，单位为秒和 g。程序导出：

- `ground_motion.csv`：`time,acceleration_g`
- `response_spectrum.csv`：目标谱、生成谱和逐点相对误差
- `time_history.png`、`response_spectrum.png`
- `settings.json`

“Smax/PGA”在参数化谱生成时表示平台谱值；“按输入 PGA 缩放”是谱匹配完成后的独立整体缩放步骤，会改变匹配谱并立即重新计算。二者不会在匹配过程中混用。

## 扩展接口

可在 `spectrum.py` 增加不同规范谱函数，在 `matching.py` 增加小波匹配器；批量随机种子筛选可重复调用 `generate_initial_motion` 和 `match_spectrum`。Arias Intensity、持时、PGV/PGD 及速度位移积分适合后续作为独立指标模块加入。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
