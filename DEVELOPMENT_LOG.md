# FlashKernel — Development Log

> This file tracks real progress, problems encountered, and solutions found.
> Updated as work happens — not retroactively.

---

## Status: NOT STARTED

### Pre-Development Research (Week 0)
- [ ] Read FlashAttention paper + blog post
- [ ] Read FlashAttention-2 paper (parallelism improvements)
- [ ] Study NVIDIA T4 Turing architecture (SM 7.5 specifics)
- [ ] Review Triton tutorial: fused attention
- [ ] Set up AWS g4dn.xlarge spot instance
- [ ] Verify CUDA toolkit version and driver on instance
- [ ] Run baseline: `torch.nn.functional.scaled_dot_product_attention` latency on T4
- [ ] Run baseline: simple PyTorch matmul benchmark to establish HBM bandwidth

---

_Entries will be added below as development progresses. Each entry includes:_
- _Date_
- _What was attempted_
- _What worked / what failed_
- _Profiling results (if any)_
- _Next steps_
