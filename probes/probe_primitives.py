"""310P 算子可用性探针：FFT 重组方案的依赖检查。

跑法（vllm-ascend 310P 容器内）:
    python probes/probe_primitives.py

预期输出（310P3, CANN 9.2, torch_npu 2.10.post7 实测）:
    ✅ matmul fp32/fp16、bmm、gather、unfold、roll、stack/cat、
       view_as_complex/view_as_real、手工实虚部复数乘、sqrt/where(实数)
    ❌ torch.fft.* / torch.stft / torch.istft（aclnn 内核缺失）
    ❌ torch.hann_window / kaiser_window（cos_ 内核缺失）
    ❌ 复数 abs/conj/where/clamp/标量乘（DT_COMPLEX64 不在支持列表）
    ❌ 复数张量 .cpu() / .to(npu) 传输（ERR01007）
"""
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
DEV = "npu:0"

results = []


def probe(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        results.append((name, "OK", ""))
    except Exception as e:
        results.append((name, "FAIL", str(e).split("\n")[0][:90]))


x = torch.randn(2, 3, 1024, device=DEV)

probe("matmul fp32", lambda: x @ torch.randn(1024, 64, device=DEV))
probe("matmul fp16", lambda: x.half() @ torch.randn(1024, 64, device=DEV, dtype=torch.float16))
probe("bmm", lambda: torch.bmm(x.unsqueeze(1), torch.randn(2, 1024, 8, device=DEV)))
probe("gather", lambda: x.gather(-1, torch.randperm(1024, device=DEV).expand(2, 3, 1024)))
probe("unfold", lambda: x.unfold(-1, 512, 128))
probe("roll", lambda: torch.roll(x, 1, -1))
probe("view_as_complex", lambda: torch.view_as_complex(torch.randn(2, 3, 513, 2, device=DEV)))
c = torch.view_as_complex(torch.randn(2, 3, 513, 2, device=DEV))
probe("complex mul (torch 内核)", lambda: c * c)
probe("手工实虚部复数乘", lambda: c.real * c.real - c.imag * c.imag)
probe("torch.fft.fft", lambda: torch.fft.fft(x))
probe("torch.stft", lambda: torch.stft(x[0, 0], n_fft=512, window=torch.hann_window(512, device="cpu").to(DEV), return_complex=True))
probe("hann_window NPU", lambda: torch.hann_window(512, device=DEV))
probe("complex abs", lambda: c.abs())
probe("复数 .cpu() 传输", lambda: c.cpu())
probe("sqrt (实数)", lambda: torch.sqrt(x * x))
probe("where (实数)", lambda: torch.where(x > 0, x, torch.zeros_like(x)))

print(f"310P 算子探针 ({torch_npu.__version__}):")
for name, st, msg in results:
    mark = "✅" if st == "OK" else "❌"
    print(f"  {mark} {name:<24} {msg}")
