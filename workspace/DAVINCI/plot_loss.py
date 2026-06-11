import re
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/kwkj-k8s/davinci/LJH/daVinci-MagiHuman2")

DEFAULT_LOG_CANDIDATES = [
    ROOT / "wandb" / "latest-run" / "files" / "output.log",
    ROOT / "train_run.log",
]

DEFAULT_OUTPUT_IMAGE = ROOT / "loss_curve.png"

def _pick_log_file(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path)
    for candidate in DEFAULT_LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_LOG_CANDIDATES[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_IMAGE))
    args = parser.parse_args()

    log_file = _pick_log_file(args.log_file)
    output_image = Path(args.output)

    steps = []
    losses = []
    v_losses = []
    a_losses = []

    try:
        with open(log_file, "r") as f:
            for line in f:
                # 寻找类似 step=4010 loss=0.4523 (v=0.3120, a=0.1403) 这样的行
                if "step=" in line and "loss=" in line:
                    step_match = re.search(r"step=(\d+)", line)
                    loss_match = re.search(r"loss=([\d.]+)", line)
                    v_match = re.search(r"v=([\d.]+)", line)
                    a_match = re.search(r"a=([\d.]+)", line)

                    if step_match and loss_match:
                        steps.append(int(step_match.group(1)))
                        losses.append(float(loss_match.group(1)))

                        if v_match:
                            v_losses.append(float(v_match.group(1)))
                        else:
                            v_losses.append(None)

                        if a_match:
                            a_losses.append(float(a_match.group(1)))
                        else:
                            a_losses.append(None)

        if not steps:
            print(f"在日志中没有找到包含 loss 的数据行：{log_file}")
            return 1

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="Total Loss", color="blue", linewidth=2)
        
        # 如果有分别的 v 和 a loss，也画出来
        if any(v is not None for v in v_losses):
            plt.plot(
                steps,
                v_losses,
                label="Video Loss",
                color="green",
                linestyle="--",
                alpha=0.7,
            )
        if any(a is not None for a in a_losses):
            plt.plot(
                steps,
                a_losses,
                label="Audio Loss",
                color="orange",
                linestyle=":",
                alpha=0.7,
            )
            
        plt.title("Training Loss Curve")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        output_image.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_image, dpi=200)
        print(f"成功！Loss 曲线图已保存为: {output_image}")
        print(f"数据来源日志: {log_file}")
        return 0
        
    except FileNotFoundError:
        print(f"找不到日志文件: {log_file}")
        return 2
    except Exception as e:
        print(f"绘制失败: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
