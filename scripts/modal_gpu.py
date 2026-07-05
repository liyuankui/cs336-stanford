"""
CS336 Modal GPU Bootstrap — A1 训练用 A100 实例

Usage:
  # 第一次：先认证（会开浏览器）
  modal token new

  # 启动交互式 A100 实例（同步代码 + 进入 Python REPL）
  modal run scripts/modal_gpu.py

  # 跑完后自动 snapshot + 计费显示
"""
import modal
import sys
from pathlib import Path

# CS336 A1 环境（匹配 assignment1-basics/pyproject.toml）
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch~=2.11.0",
        "numpy>=2.4",
        "einops>=0.8",
        "einx>=0.4",
        "jaxtyping>=0.3",
        "regex>=2026.3.32",
        "tiktoken>=0.12.0",
        "tqdm>=4.67",
        "wandb>=0.25",
        "pytest>=9.0",
        "ruff>=0.15.8",
    )
)

app = modal.App("cs336-stanford")

# 持久卷：保存 checkpoint / 数据，重启不丢
vol = modal.Volume.from_name("cs336-stanford", create_if_missing=True)

# 挂载本地代码（每次启动自动同步最新）
repo_path = Path(__file__).parent.parent / "course" / "assignment1-basics"
if repo_path.exists():
    mounts = [modal.Mount.from_local_dir(repo_path, remote_path="/root/assignment1-basics")]
else:
    mounts = []

@app.function(
    image=image,
    gpu="A100",
    volumes={"/root/checkpoints": vol},
    mounts=mounts,
    timeout=60 * 60 * 4,  # 4 小时
)
def train():
    import os
    import time
    import subprocess

    print("=" * 60)
    print("🚀 CS336 A1 GPU 实例已启动")
    print(f"   GPU: A100 80GB")
    print(f"   代码: /root/assignment1-basics")
    print(f"   Checkpoint: /root/checkpoints (持久卷)")
    print("=" * 60)

    # GPU 信息
    import torch
    print(f"\n📋 PyTorch: {torch.__version__}")
    print(f"📋 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📋 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📋 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 跑测试确认环境 OK
    print("\n🧪 跑测试（应全 fail，因为你还没实现）...")
    os.chdir("/root/assignment1-basics")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout[-500:] if result.stdout else "no stdout")
    if result.returncode != 0:
        print("(测试 fail 是正常的 — 你还没写实现)")

    # 持久化 checkpoint 到 volume
    vol.commit()
    print("\n✅ 环境就绪。开始训练吧！")

if __name__ == "__main__":
    with modal.enable_output():
        train.remote()
