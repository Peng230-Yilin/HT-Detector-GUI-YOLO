# HT-Detector-GUI-YOLO

当前产品版本：**1.0.0**

HT-Detector-GUI-YOLO 是面向比色皿图像研究场景的 Windows 桌面工具，使用固定的 YOLOv8n 检测模型定位比色皿和液相区域，并支持颜色数据提取、线性回归、浓度检测、绘图与结果保存。

## v1.0.0 功能范围

- Import Image：导入线性回归用的标定图像。
- Linear Regression：识别标定样品并计算 RGB 通道的线性回归结果。
- Plot：显示所选通道的标准曲线、公式和 R²。
- Detect：识别待测图像并根据当前回归公式计算浓度。
- Save：保存线性回归或检测结果。
- Camera Start/Stop：按需启动或停止实时摄像头预览。

摄像头默认关闭，新建附加窗口不会占用摄像头。v1.0.0 只提供最小 Start/Stop 实时预览；拍照、录像、曝光、白平衡及其他详细相机设置计划在 v1.3 实现。

## 已验证环境

- Windows
- Python 3.10.11（64 位）
- CPU 推理；不要求 CUDA
- PyTorch 2.13.0+cpu
- TorchVision 0.28.0+cpu
- 仓库内 Ultralytics 8.1.47 源码

## 从零安装

以下命令在仓库根目录的 PowerShell 中执行。需要能够访问 PyPI 和 PyTorch 官方 CPU wheel 索引。

### 自动安装

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install-v1.0.0.ps1
```

脚本会依次尝试由 Windows Python Launcher 的 `py -3.10`、PATH 中的 `python` 和 PATH 中的 `python3.10` 定位解释器。已安装 Python 并不保证 Python Launcher 已完成注册；若 Launcher 无法发现 Python，可直接指定 `python.exe`，不需要卸载或重新安装现有 Python：

```powershell
.\install-v1.0.0.ps1 -PythonExecutable "C:\Path\To\Python310\python.exe"
```

指定路径后，脚本只使用该路径且不会静默回退。无论自动发现还是显式指定，解释器都必须通过 CPython 3.10.11、64 位和实际 `sys.executable` 路径验证。脚本随后创建 `.venv`，先从 `https://download.pytorch.org/whl/cpu` 安装经过验证的 CPU Torch 组合，再安装 GUI 运行依赖，最后以 editable、`--no-deps` 方式安装仓库内 `HT-Detector_Peng`。脚本拒绝覆盖已有 `.venv`。

### 手动安装

```powershell
$PythonExecutable = "C:\Path\To\Python310\python.exe"
& $PythonExecutable -m venv .venv
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -B -m pip install --upgrade pip
.venv\Scripts\python.exe -B -m pip install -r requirements-torch-cpu.txt
.venv\Scripts\python.exe -B -m pip install -r requirements-runtime.txt
.venv\Scripts\python.exe -B -m pip install --no-deps --editable .\HT-Detector_Peng
.venv\Scripts\python.exe -B -m pip check
```

必须安装仓库内 `HT-Detector_Peng`，并保持最后一步使用 `--no-deps --editable`；不要再执行 `pip install ultralytics`，否则可能覆盖或绕过本项目的可信权重加载修改。

## 启动

在仓库或运行包根目录执行：

```powershell
.venv\Scripts\python.exe -B Peng1.0_GUI\main.py
```

## Linear Regression 操作顺序

1. 点击 **Import Image**，选择包含已知浓度标定样品的图像。
2. 在界面中确认或输入标定浓度设置。
3. 点击 **Linear Regression**，等待识别和回归完成。
4. 点击 **Plot** 查看当前颜色通道的标准曲线、回归公式和 R²。
5. 点击 **Save Linear** 保存当前结果。

默认标定图片目录：

```text
HT-Detector_Peng/custom/linear_detection/linear
```

线性回归保存目录：

```text
HT-Detector_Peng/runs/detect/results/linear
```

保存内容包括：

- `linear_con_rgb.xlsx`：标定样品和 RGB 通道回归公式
- `standard_curve.png`：标准曲线
- `calibration_annotated.png`：标注后的标定图像

## Detection 操作顺序

1. 先完成一次 Linear Regression；未保存的当前回归公式也可供本次检测使用。
2. 点击 **Detect**，选择待测图片。
3. 等待识别、RGB 提取和浓度计算完成。
4. 检查检测图像与结果表。
5. 点击 **Save Detection** 保存当前结果。

默认检测图片目录：

```text
HT-Detector_Peng/custom/linear_detection/detection
```

检测保存目录：

```text
HT-Detector_Peng/runs/detect/results/detection
```

每次保存生成同名的一组文件：

- `.xlsx`：样品编号、浓度、RGB 与 ROI 信息
- `.png`：标注后的检测图像

`runs/` 是运行时输出目录，不属于正式运行包的受控源文件。

## 摄像头

主窗口中的摄像头初始状态为关闭。点击 **Start Camera** 开始实时预览，点击 **Stop Camera** 释放设备。附加窗口不提供摄像头访问。拍照、录像和相机参数控制不属于 v1.0.0。

## 固定模型与安全说明

固定权重路径：

```text
HT-Detector_Peng/weights/cuvette_Peng/yolov8n_train/weights/best.pt
```

模型的用途、来源、大小、SHA-256 和限制见 [MODEL_CARD.md](MODEL_CARD.md)。加载器会校验固定路径、文件大小和 SHA-256。

不要使用任意来源的 `.pt` 替换固定权重。PyTorch checkpoint 可能包含可执行的序列化对象；未经验证的权重不应被加载。

## v1.0.0 已知限制

- 仅针对当前比色皿研究场景和已验证图像条件；其他场景需要重新验证。
- 仅支持固定的 `best.pt`，不支持任意模型切换。
- 当前发布基线是 Windows、Python 3.10.11 和 CPU 推理。
- 摄像头只提供 Start/Stop 预览。
- 界面尚未提供完整中英文切换。
- 结果默认写入运行包目录下的 `HT-Detector_Peng/runs/`；请确保目录可写并自行备份结果。

## 许可证与第三方组件

项目原创代码和修改的权利人为 Peng Yilin，并按 AGPL-3.0 发布。固定模型 `best.pt` 的共同训练者和权利人为 Peng Yilin、Yue Hengmao。第三方代码、Ultralytics 基础模型和资源仍保留各自许可证及权利声明。

- 完整许可证：[LICENSE](LICENSE)
- 第三方组件和历史 MIT 组件说明：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- 许可证文件映射：[LICENSES/README.md](LICENSES/README.md)

现有 `Peng1.0_GUI/LICENSE` 和 `HT-Detector_Peng/LICENSE` 作为原始 hmy-repo 组件的 MIT 许可证保留；这不表示撤销或追溯更改历史 MIT 授权。

## 发布命名

- 产品版本：`1.0.0`
- 计划 Git tag：`v1.0.0`
- Release 标题：`HT-Detector-GUI-YOLO v1.0.0`
- 运行包名称：`HT-Detector-GUI-YOLO-v1.0.0`
