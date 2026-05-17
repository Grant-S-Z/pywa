# PyWA - Python Waveform Analysis

高性能 PMT 波形分析工具，用于从光电倍增管 (PMT) 波形中重建光电子 (PhotoElectron, PE) 的幅度和时间信息。

## 项目简介

PyWA 是一个专为粒子物理实验中 PMT 波形分析设计的工具包，支持在 GPU 上高效处理批量波形数据。主要功能包括：

- **波形预处理**：基线扣除、噪声估计
- **反卷积**：Richardson-Lucy (RL) 反卷积提取初始 PE 信息
- **似然拟合**：结合时间和电荷信息的联合似然函数优化
- **可视化**：丰富的绘图工具用于质量控制和结果展示

## 物理原理

### 单光电子响应 (SER)

PMT 的单光电子响应可以用以下公式描述：

$$
\text{SER}(t) = A_0 \cdot \left[\frac{1}{\tau} \exp\left(-\frac{t}{\tau}\right) \theta(t)\right] \otimes \left[\frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{t^2}{2\sigma^2}\right)\right]
$$

其中：
- $A_0$: 单光电子电荷
- $\tau$: 指数衰减时间常数
- $\sigma$: 高斯展宽参数
- $\theta(t)$: 阶梯函数

### 波形模型

观测到的波形是多个光电子响应的叠加加上噪声：

$$
w(t) = \sum_{i=1}^{N_{\text{PE}}} q_i \cdot \text{SER}(t - t_i) + \varepsilon(t)
$$

### 似然函数

拟合采用联合似然函数：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \mathcal{L}_{\text{charge}}
$$

- **时间似然**：
  $$\mathcal{L}_{\text{time}} = \sum_{t=t_{\text{begin}}}^{t_{\text{end}}} \log \mathcal{N}\left(V_{\text{obs}}(t) - V_{\text{fit}}(t) \mid 0, \sigma_{\text{baseline}}\right)$$
- **电荷似然**：
  $$\mathcal{L}_{\text{charge}} = \sum_{i=1}^{N} \log \mathcal{N}(Q_i \mid \text{SPE model})$$

## 项目结构

```
pywa/
├── wavelike/               # 核心分析模块
│   ├── __init__.py
│   ├── config.py          # 全局配置参数
│   ├── io.py              # ROOT 文件读写
│   ├── pmtparam.py        # PMT 参数管理
│   ├── preprocess.py      # 波形预处理
│   ├── physics.py         # 物理模型（SER）
│   ├── likelihood.py      # 似然函数定义
│   ├── fit.py             # 波形拟合
│   ├── plot.py            # 可视化工具
│   └── utils.py           # 工具函数
├── data/                  # 数据文件
│   ├── DN_fit_results_simple.csv    # PMT 增益参数
│   ├── LSGainList_fsmp.csv          # PMT 增益列表
│   ├── TimeCalib.csv                # 时间刻度
│   └── run00045887/                 # 实验数据 (ROOT)
├── plots/                 # 输出图像
├── output/                # 输出数据
├── log/                   # 日志文件
├── main.py                # 批处理分析主程序
├── main_serial.py         # 串行分析主程序
└── main_serial_search.py  # 参数搜索程序
```

## 安装依赖

```bash
pip install torch numpy scipy matplotlib uproot awkward iminuit numba
```

## 快速开始

### 1. 准备数据

确保以下数据文件位于 `data/` 目录：
- PMT 增益参数文件：`DN_fit_results_simple.csv`, `LSGainList_fsmp.csv`
- 时间校准文件：`TimeCalib.csv`
- 实验数据文件（ROOT 格式）

### 2. 配置参数

编辑 `wavelike/config.py` 设置全局参数：

```python
# 波形参数
window_size = 900      # 波形窗口大小
ser_length = 50        # SER 长度
template_range = 50    # 模板范围
max_pe = 200          # 最大 PE 数

# 基线参数
bl_begin = 0
bl_end = 100

# 积分窗口
inte_begin = 100
inte_end = 700
```

### 3. 运行分析

#### 批处理模式（GPU）

```bash
python main.py
```

适合大规模数据处理，使用 GPU 加速。

#### 串行模式

```bash
python main_serial.py
```

逐事件处理，包含详细的中间步骤可视化，适合调试和质量控制。

## 使用示例

### 加载 PMT 参数

```python
from wavelike import load_all_pmt_params

pmt_params = load_all_pmt_params(
    dn_csv='data/DN_fit_results_simple.csv',
    gain_csv='data/LSGainList_fsmp.csv',
    gauss_csv='data/1.csv',
    time_csv='data/TimeCalib.csv',
    template_range=50,
    gauss_no=5
)
```

### 读取波形数据

```python
from wavelike.io import DataReader

data_reader = DataReader(
    'data/run00045887/Jinping_1ton_Phy_20250302_00045887.root',
    allowed_pmts=set(pmt_params.keys())
)

for ids_batch, waveform_batch in data_reader.get_batch_generator(batch_size=32):
    # 处理批量波形
    pass
```

### 波形预处理

```python
from wavelike.preprocess import preprocess_waveform_batch

waveform_batch, waveform_norm_batch, deconv_batch, pe_prior_batch, cpe_batch, noise_batch = \
    preprocess_waveform_batch(
        waveform_batch, 
        ids_batch, 
        pmt_params, 
        device='cuda'
    )
```

### 波形拟合

```python
from wavelike.fit import WaveformFitter

fitter = WaveformFitter(waveform, pmt_param, noise_sigma)
fit_result = fitter.fit(n_pe=5)
```

## 输出说明

### 日志文件

程序运行日志保存在 `log/` 目录：
- `main_serial.log`: 串行分析日志

### 图像输出

各类分析图像保存在 `plots/` 目录：
- `original_plots/`: 原始波形
- `ser_plots/`: 单光电子响应
- `sub_plots/`: 扣除基线后波形
- `deconv_plots/`: 反卷积结果
- `fit_plots/`: 拟合结果
- `prior_comparison_plots/`: 先验比较

### ROOT 输出

拟合结果保存为 ROOT 文件在 `output/` 目录：
- 包含重建的 PE 时间、幅度等信息

## 性能优化

### GPU 加速

支持 CUDA 加速，大幅提升批处理性能：

```python
device = 'cuda'  # 'cpu', 'mps'
```

### Numba JIT 编译

关键计算函数使用 Numba JIT 加速：
- SER 波形计算
- 反卷积迭代

### 批处理优化

- 使用 `batch_size` 参数调整批大小以平衡内存和速度


---

**更新日期**: 2025-12-18
