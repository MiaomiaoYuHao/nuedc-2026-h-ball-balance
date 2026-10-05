# 2026 全国大学生电子设计竞赛 H 题 · 球杆平衡控制系统

> 参赛代码，**获三等奖**。一套在树莓派上运行的球杆平衡（ball & beam）闭环系统：
> 摄像头识别轨道上的小球位置，控制器解算目标倾角，步进电机经曲柄抬降横梁，
> 绝对式编码器作为内环反馈。

## 现场效果

- 输入：Picamera2 采集的轨道画面（640×480 @ 60 fps）
- 检测：ROI 列剖面 → 空轨基线差分 → 零均值匹配滤波 → SNR 门限 → Kalman 跟踪
- 控制：外环 P+D（位置误差 → 目标倾角），内环用步进电机 + 绝对编码器的角度闭环
- 安全：绝对编码器的物理安全窗口，独立于软件零点，避免撞机械限位

## 硬件

| 部件 | 型号 / 说明 |
| --- | --- |
| 主控 | 树莓派 5（GPIO 经 RP1，需 `lgpio`） |
| 相机 | Raspberry Pi Camera（Picamera2 / libcamera） |
| 执行器 | MS42CG 步进电机（1.8°/步），16 细分 |
| 驱动器 | WHEELTEC D36A / A4988 / DRV8825 / TMC2209（使能极性在 `config.py` 中可切换） |
| 反馈 | MS42CG 内置 MT6816 磁编码器，取 **PWM 绝对角度**输出 |
| 机构 | 曲柄半径 30 mm 抬升 500 mm 横梁 |

接线约定见 `config.py` 的 `pin_*` 注释（已避开 I2C / SPI / UART 引脚）。

## 安装

树莓派 OS (Bookworm) 上：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-lgpio python3-picamera2
# 或使用 pip（picamera2/lgpio 建议走 apt 源）
pip install -r requirements.txt
```

## 运行

本仓库是一个 Python 包，用 `-m` 方式运行。包名必须是合法的 Python 标识符，
克隆时给它一个合法名字即可（推荐 `hball`）：

```bash
git clone https://github.com/lina130/nuedc-2026-h-ball-balance.git hball
cd hball/..            # 回到 hball 的上一级目录
python -m hball --help
```

上电后建议**按顺序**做三步自检（不启动相机）：

```bash
python -m hball --enc-test     # 编码器读数是否正常
python -m hball --cal-steps    # 用编码器实测驱动器真实细分
python -m hball --motor-test   # 电机扫摆、能否保持位置
```

自检通过后跑主程序：

```bash
python -m hball                     # 带 GUI（可调参、可看检测画面）
python -m hball --headless          # 无窗口，帧率最高
python -m hball --no-motor          # 只跑视觉，不动电机（安全调试）
```

## 键盘操作

| 键 | 作用 | 键 | 作用 |
| --- | --- | --- | --- |
| `q` | 退出 | `空格` | 启停控制 |
| `p` | 切换 PID / bang-bang | `c` | 设定当前水平零点 |
| `r` | 框选 ROI | `b` | 重采空轨基线 |
| `v` | 保存参数 | `o` | 使能/断开驱动器 |
| `t` | 重置跟踪器 | `j` | 方向自检 |
| `e` | 自动曝光开关 | `E` / `w` | 曝光 +/− |
| `[` / `]` | 目标位置左右移 | `a` / `d` | 零点微调 |
| `m` | 切换视图 | `x` | 切换配色 |
| `g` | 显示/隐藏控制面板 | `l` | 记录 CSV 日志 |
| `z` | 导出诊断快照 | | |

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `cli.py` | 入口：参数解析与模式分发 |
| `app.py` | 主循环：采集 → 检测 → 控制 → 绘制 → 按键 |
| `config.py` | 全部可调参数（`Settings` 数据类，GUI 滑块直接写它） |
| `camera.py` | 后台采集线程与曝光控制 |
| `vision.py` | 球检测：匹配滤波 → SNR 门限 → Kalman 跟踪 |
| `controller.py` | 外环位置控制器（PID / bang-bang） |
| `kinematics.py` | 曲柄—横梁几何换算与编码器安全窗口 |
| `motor.py` | 步进电机步进线程、位置目标与限幅 |
| `encoder.py` | MT6816 绝对角度读取 |
| `gpio_backend.py` | `lgpio` 兼容层（Pi 5 必需） |
| `diagnostics.py` | 上电自检：编码器 / 细分 / 电机 |
| `diagrec.py` | 逐帧诊断记录与转储 |
| `ui.py` | 窗口、手绘控制面板、覆盖层绘制 |

## 几个值得说明的设计

- **为什么用编码器的 PWM 而不是 A/B 相**：A/B 正交后 4096 计数/转，曲柄摆动时约 5 kHz 边沿，
  Python 处理必然丢计数，而丢计数会造成**永久**角度误差；PWM 每帧都输出绝对角度（约 1 kHz），
  漏一帧不影响下一帧的正确性，且断电后仍准确。
- **为什么不用 RPi.GPIO**：Pi 5 把 GPIO 换成独立 RP1 芯片，RPi.GPIO 会报
  `Cannot determine SOC peripheral base address`；`lgpio` 直接走内核 gpiochip 接口，无需守护进程。
- **为什么匹配滤波用零均值核**：零均值可抵消均匀/缓变光照，自动曝光"呼吸"和阴影都不会产生假峰。
- **为什么工作在用 SNR 而不是灰度阈值**：SNR 无量纲且自校准，换了光照阈值依然成立。
- **KD 为什么一定要低通**：接近目标时速度信号几乎全是跟踪器噪声，KD 会把它放大成电机抖动。
- **积分项为什么是 0**：被控对象本身不稳定，积分只会让收敛更难。

## 参数调节

全部参数集中在 `config.py` 的 `Settings`。最常见的两件事：

- **球越跑越偏而不是回中** → 把 `motor_dir_sign` 取反（或运行时按 `j`）。
- **电机完全不励磁** → `enable_active_low` 与驱动器不匹配，见 `config.py` 注释里的对照表。

## 许可

MIT License，见 [LICENSE](LICENSE)。