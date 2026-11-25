# npu-fft-310p

**Pure-PyTorch FFT/STFT re-implementation for Huawei Ascend 310P (Atlas 300I Duo) — no custom kernels, only verified on-device operators.**

**昇腾 310P（Atlas 300I Duo）上的纯 PyTorch FFT/STFT 重组实现 —— 无自定义内核，只用真机已验证的算子。**

[English](#english) | [简体中文](#简体中文)

---

<a name="english"></a>

## English

### The problem

Ascend 310P (Atlas 300I Duo, `soc=ascend310p`) is an **inference** card. Its CANN operator library
does not ship the FFT family of kernels — `torch.fft.*`, `torch.stft`, `torch.istft` all fail at
runtime with `NPU function error` / `aclStft failed (161002)`, and even `torch.hann_window` fails
(`cos_` kernel missing). The official torch_npu API docs only list **Atlas A2 (910B), A3 (910C),
Ascend 950DT** as supported for the whole `torch.fft` module; 310P is simply absent.

This blocks every speech model on 310P: TTS front-ends (IndexTTS, Qwen3-TTS, MiniCPM-o),
vocoder pre/post-processing, ASR feature extraction — all need STFT.

### The solution

Notably, **Ascend 950DT also lacked `torch.stft`** (it has `torch.fft.*` though) — vllm-omni
[PR #9869](https://github.com/vllm-project/vllm-omni/pull/9869) rebuilt STFT from `rfft` for A5.
310P is one level deeper: it has **no FFT at all**, so we rebuild the FFT itself.

This library re-implements the FFT family using only operators that a 70-operator on-device probe
confirmed working on 310P3 (matmul, gather, unfold, roll, stack/cat, view_as_real/complex,
elementwise mul/add/sqrt):

| Component | Method | Accuracy vs CPU (rel) |
|---|---|---|
| `fft` / `ifft`, N = 2^k | **Radix-2 iterative Cooley–Tukey** (bit-reversal via `gather`, butterflies via elementwise ops) | ~1.5e-07 |
| `fft` / `ifft`, any N | **Bluestein's algorithm** (chirp-z transform: any-N DFT → linear convolution → radix-2 FFT) | ~3e-07 |
| `rfft` / `irfft` | One-sided spectrum + conjugate-symmetric reconstruction (real/imag split, since 310P lacks complex `conj`) | ~3e-07 |
| `stft` | reflect-pad + `unfold` framing + windowing + radix-2 FFT | ~1.4e-07 |
| `istft` | per-frame irfft + overlap-add (real-domain normalization; 310P lacks complex `clamp`/`where`) | ~1.3e-07 |
| `hann` / `hamming` / `kaiser_window` | generated on CPU once, cached to NPU | exact |

Complex tensors are only materialized at function boundaries. All intermediate math keeps
real/imaginary parts as separate fp32 tensors — this works around the 310P's missing complex
operator kernels (`abs`, `conj`, `where`, `clamp`, scalar-mul on complex are all unsupported) and
the fact that complex tensors cannot be transferred NPU↔CPU at all (`ERR01007`).

### Verified on

- Atlas 300I Duo (Ascend 310P3, 44GB), npu-smi 25.5.1
- Container `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p` (CANN 9.2.0)
- torch 2.10.0 + torch_npu 2.10.0.post7, Linux aarch64, Python 3.12

### Quick start

```bash
git clone https://github.com/green-hand-shandong/npu-fft-310p.git
cd npu-fft-310p

# inside a vllm-ascend 310P container with an NPU attached:
python probes/probe_primitives.py   # 1. confirm operator availability
python tests/test_npu_fft.py        # 2. 13 correctness tests, all vs CPU reference
```

Use as a drop-in library:

```python
import npu_fft

spec = npu_fft.stft(waveform_npu, n_fft=1024, hop_length=256,
                    window=npu_fft.hann_window(1024, device="npu:0"),
                    center=True, pad_mode="reflect")     # complex64, matches torch.stft
wav  = npu_fft.istft(spec, n_fft=1024, hop_length=256, center=True)

spec_any = npu_fft.fft(x, n=500)                          # any N via Bluestein
```

### Wiring into your model

Monkey-patch at startup (same approach as `vllm-omni` PR #9869 for A5):

```python
import torch, npu_fft
torch.stft = npu_fft.stft
torch.istft = npu_fft.istft
torch.hann_window = npu_fft.hann_window
torch.kaiser_window = npu_fft.kaiser_window
import torch.fft as _tfft
for _n in ("fft", "ifft", "rfft", "irfft"):
    setattr(_tfft, _n, getattr(npu_fft, _n))
```

### End-to-end validation

IndexTTS-2.5 voice-cloning TTS on a single 310P3 chip, mel front-end switched from the
CPU fallback to this NPU-native STFT:

| Metric | CPU fallback (before) | npu-fft (after) |
|---|---|---|
| mel front-end (15s ref audio) | 25.8 ms | **17.2 ms** (−33%) |
| log-mel feature diff | — | max 4e-4 (model runs fp16, noise floor ~1e-2) |
| E2E synthesis | RTF 5.93 | RTF 5.86 (unchanged) |

### Performance notes

- Short-frame STFT (n_fft=1024, 15s×2 audio): NPU 51 ms vs CPU 20 ms — the butterfly loop is
  Python-level; throughput work (fused scatter for bit-reversal, `fold`-based overlap-add) is
  welcome. The win today is **zero H2D/D2H round-trips** when the rest of the model lives on NPU.
- Non-power-of-2 sizes go through Bluestein (3 FFTs of the next pow2); prefer pow2 n_fft.
- 310P fp32 `matmul` shows ~350× worse accumulation error than CPU (suspected low-precision
  internal accumulation). Direct DFT-matrix multiplication is therefore **not** viable — that is
  exactly why Bluestein (no large-K reduction) is the default for arbitrary N.

### Operator findings on 310P (reusable beyond this repo)

| Operator | Status | Note |
|---|---|---|
| `torch.fft.*`, `stft`, `istft` | ❌ missing | aclnn kernels not shipped for 310P |
| `hann_window`, `kaiser_window` | ❌ missing | `cos_` kernel missing |
| complex `abs`/`conj`/`where`/`clamp`/scalar-mul | ❌ missing | `DT_COMPLEX64` not in support list |
| complex tensor H2D/D2H transfer | ❌ fails | `ERR01007` — split to real/imag first |
| fp32 `matmul` accuracy | ⚠️ degraded | ~350× CPU error; fp16/bmm/blocked-K do not recover |
| matmul/bmm/gather/unfold/roll/elementwise | ✅ | full-speed workhorses used here |

### License

Apache-2.0. See [LICENSE](LICENSE).

---

<a name="简体中文"></a>

## 简体中文

### 问题背景

昇腾 310P（Atlas 300I Duo）是**推理卡**，CANN 算子库没有提供 FFT 家族内核——`torch.fft.*`、
`torch.stft`、`torch.istft` 在真机上全部报 `NPU function error` / `aclStft failed (161002)`，
连 `torch.hann_window` 都不可用（`cos_` 内核缺失）。官方 torch_npu 文档的 `torch.fft` 支持
清单只列了 **Atlas A2（910B）、A3（910C）、Ascend 950DT**，310P 整个产品线缺席。

这直接堵死了 310P 上的所有语音模型：TTS 前端（IndexTTS、Qwen3-TTS、MiniCPM-o）、声码器
前后处理、ASR 特征提取——全都依赖 STFT。

### 解决方案

有意思的是，**950DT 也缺 `torch.stft`**（但有 `torch.fft.*`）——vllm-omni
[PR #9869](https://github.com/vllm-project/vllm-omni/pull/9869) 在 A5 上用 `rfft` 重组了 STFT。
310P 更深一层：**连 FFT 都没有**，所以我们把 FFT 本体重建了。

本库只用真机 70 算子探针确认可用的算子（matmul、gather、unfold、roll、stack/cat、
view_as_real/complex、逐元素乘加/sqrt）重组了整个 FFT 家族：

| 组件 | 方法 | 对比 CPU 精度（rel） |
|---|---|---|
| `fft` / `ifft`（N=2^k） | **radix-2 迭代蝶形**（`gather` 做 bit-reversal，逐元素算子做蝶形） | ~1.5e-07 |
| `fft` / `ifft`（任意 N） | **Bluestein 算法**（chirp-z：任意 N 的 DFT → 线性卷积 → radix-2 FFT） | ~3e-07 |
| `rfft` / `irfft` | 单边谱 + 共轭对称重建（实虚分离，绕开 310P 缺失的复数 `conj`） | ~3e-07 |
| `stft` | reflect-pad + `unfold` 分帧 + 加窗 + radix-2 FFT | ~1.4e-07 |
| `istft` | 逐帧 irfft + overlap-add（实数域归一化；310P 缺复数 `clamp`/`where`） | ~1.3e-07 |
| `hann` / `hamming` / `kaiser_window` | CPU 生成一次后缓存到 NPU | 精确一致 |

复数张量只在函数边界组装。所有中间计算都用实/虚两个 fp32 张量表示——这同时绕开了
310P 缺失的复数算子内核（`abs`/`conj`/`where`/`clamp`/复数标量乘全部不支持），以及
复数张量完全无法在 NPU↔CPU 间传输的限制（`ERR01007`）。

### 真机验证环境

- Atlas 300I Duo（Ascend 310P3，44GB），npu-smi 25.5.1
- 容器 `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p`（CANN 9.2.0）
- torch 2.10.0 + torch_npu 2.10.0.post7，Linux aarch64，Python 3.12

### 快速开始

```bash
git clone https://github.com/green-hand-shandong/npu-fft-310p.git
cd npu-fft-310p

# 在挂载 NPU 的 vllm-ascend 310P 容器内:
python probes/probe_primitives.py   # 1. 确认算子可用性
python tests/test_npu_fft.py        # 2. 13 项正确性测试，全部对标 CPU 参考值
```

作为库直接使用：

```python
import npu_fft

spec = npu_fft.stft(waveform_npu, n_fft=1024, hop_length=256,
                    window=npu_fft.hann_window(1024, device="npu:0"),
                    center=True, pad_mode="reflect")     # complex64，与 torch.stft 对齐
wav  = npu_fft.istft(spec, n_fft=1024, hop_length=256, center=True)

spec_any = npu_fft.fft(x, n=500)                          # 任意 N 走 Bluestein
```

### 接入你的模型

启动时 monkey-patch（与 vllm-omni PR #9869 在 A5 上的做法一致）：

```python
import torch, npu_fft
torch.stft = npu_fft.stft
torch.istft = npu_fft.istft
torch.hann_window = npu_fft.hann_window
torch.kaiser_window = npu_fft.kaiser_window
import torch.fft as _tfft
for _n in ("fft", "ifft", "rfft", "irfft"):
    setattr(_tfft, _n, getattr(npu_fft, _n))
```

### 端到端验证

IndexTTS-2.5 声音克隆 TTS，单颗 310P3，mel 前端从 CPU fallback 切换到本库 NPU 原生 STFT：

| 指标 | CPU fallback（切换前） | npu-fft（切换后） |
|---|---|---|
| mel 前端（15s 参考音频） | 25.8 ms | **17.2 ms**（−33%） |
| log-mel 特征差异 | — | max 4e-4（模型 fp16 推理噪声 ~1e-2） |
| 端到端合成 | RTF 5.93 | RTF 5.86（持平） |

### 性能说明

- 短帧 STFT（n_fft=1024，15s×2 音频）：NPU 51 ms vs CPU 20 ms——蝶形循环是 Python 层的；
  吞吐优化（bit-reversal 融合 scatter、`fold` 版 overlap-add）欢迎 PR。当前收益是
  **零 H2D/D2H 往返**（模型其余部分都在 NPU 上时）。
- 非 2 次幂尺寸走 Bluestein（下一个 2 次幂的 3 次 FFT）；优先使用 2 次幂 n_fft。
- 310P 的 fp32 `matmul` 实测累加误差比 CPU 差 ~350 倍（疑似内部低精度累加）。直接
  DFT 矩阵乘因此**不可行**——这正是任意 N 默认走 Bluestein（无大 K 归约）的原因。

### 310P 算子结论（可复用，超出本库范围）

| 算子 | 状态 | 说明 |
|---|---|---|
| `torch.fft.*`、`stft`、`istft` | ❌ 缺失 | 310P 未随 CANN 发布 aclnn 内核 |
| `hann_window`、`kaiser_window` | ❌ 缺失 | `cos_` 内核缺失 |
| 复数 `abs`/`conj`/`where`/`clamp`/标量乘 | ❌ 缺失 | `DT_COMPLEX64` 不在支持列表 |
| 复数张量 H2D/D2H 传输 | ❌ 失败 | `ERR01007`——先拆实虚再搬 |
| fp32 `matmul` 精度 | ⚠️ 降级 | ~350× CPU 误差；fp16/bmm/分块 K 均无法恢复 |
| matmul/bmm/gather/unfold/roll/逐元素 | ✅ | 本库的高速基石 |

### 许可证

Apache-2.0，见 [LICENSE](LICENSE)。

### 致谢

- [index-ascend310p](https://github.com/green-hand-shandong/index-ascend310p) — IndexTTS-2.5 × 310P 移植，本库的算子探针与真机验证环境来自该项目
- [vllm-omni PR #9869](https://github.com/vllm-project/vllm-omni/pull/9869) — 950DT (A5) 上用 `rfft` 重组 STFT 的先行思路
- [vllm-ascend](https://github.com/vllm-project/vllm-ascend) — 310P 推理栈
