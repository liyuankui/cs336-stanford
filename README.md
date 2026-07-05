# CS336: Language Modeling from Scratch

> Stanford CS336 自学手搓 LLM 项目 · Kyle Li · 2026-07 启动

## 关于

这是 [Stanford CS336](https://cs336.stanford.edu/) 的自学实现仓库。
5 个 assignment，全部从零实现（BPE → Transformer → Systems → Scaling → Data → Alignment）。

- **学习契约**: [`../Notebooks/workspace/stanford-cs336/SPEC.md`](../Notebooks/workspace/stanford-cs336/SPEC.md)
- **执行手册**: [`../Notebooks/workspace/stanford-cs336/OPS.md`](../Notebooks/workspace/stanford-cs336/OPS.md)
- **进度追踪**: [`../Notebooks/workspace/stanford-cs336/PROGRESS.md`](../Notebooks/workspace/stanford-cs336/PROGRESS.md)

## 目录结构

```
cs336-stanford/
├── assignment1-basics/      # BPE + Transformer + 训练
├── assignment2-systems/     # Flash Attention 2 + profiling
├── assignment3-scaling/     # Scaling laws 实验
├── assignment4-data/        # Common Crawl 清洗
├── assignment5-alignment/   # SFT + DPO
└── notes/                   # 每个 assignment 的学习笔记
```

## 环境

- Python 3.11+ / PyTorch 2.x
- GPU: Modal (A100, $30 free credit)
- 每个 assignment 子目录有自己的 `pyproject.toml`

## 进度

见 [PROGRESS.md](https://github.com/liyuankui/cs336-stanford/blob/main/PROGRESS.md)

## License

MIT (学习目的，引用 Stanford CS336 课程内容版权归原作者)
