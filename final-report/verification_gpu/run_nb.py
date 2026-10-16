"""Execute a notebook's code cells in one fresh interpreter and save logs/figures.

The Windows raw-export path is redirected into this run's output directory.
Scientific cells are otherwise unchanged.
"""
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/khuong/workspace/Trustworthy_AI_Project")


def main():
    nb_path = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    os.chdir(nb_path.parent)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    nb = json.loads(nb_path.read_text())
    ns = {"__name__": "__main__"}
    log_lines = []
    cell_status = []

    class Tee(io.TextIOBase):
        def __init__(self, stream):
            self.stream = stream
            self.buf = io.StringIO()

        def write(self, s):
            self.stream.write(s)
            self.stream.flush()
            self.buf.write(s)
            return len(s)

        def flush(self):
            self.stream.flush()

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        src = "\n".join(
            line for line in src.splitlines()
            if not line.startswith("%") and not line.startswith("!")
        )
        src = src.replace(
            r'RAW_DIR = r"E:\Project\Trustworthy_AI_Project\final-report\raw"',
            f'RAW_DIR = r"{out_dir}"',
        )
        src = src.replace(
            "Path.cwd().parent / 'final-report' / 'verification_cpu'",
            f"Path(r'{out_dir}')",
        )
        stdout, stderr = Tee(sys.stdout), Tee(sys.stderr)
        print(f"\n===== CELL {i} =====", flush=True)
        err = None
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile(src, f"{nb_path.name}:cell{i}", "exec"), ns, ns)
        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
        log_lines.append(f"\n===== CELL {i} =====\n")
        log_lines.append(stdout.buf.getvalue())
        err_txt = stderr.buf.getvalue()
        if err_txt:
            log_lines.append(err_txt)
        nfig = 0
        for num in plt.get_fignums():
            fig = plt.figure(num)
            dest = fig_dir / f"cell{i:02d}_fig{nfig}.png"
            fig.savefig(dest, bbox_inches="tight")
            nfig += 1
        plt.close("all")
        if "test_mse" in ns and "test_psnr" in ns and not ns.get("_logged_test_mse"):
            mse = ns["test_mse"]
            psnr = ns["test_psnr"]
            fmse = ns.get("feat_mse")
            print(f"SNAPSHOT mse={float(mse):.6f} psnr={float(psnr):.4f} fmse={float(fmse) if fmse is not None else float('nan'):.6e}", flush=True)
            ns["_logged_test_mse"] = True
        cell_status.append({"cell": i, "figures": nfig, "error": err})
        (out_dir / "log.txt").write_text("".join(log_lines))
        (out_dir / "cells.json").write_text(json.dumps(cell_status, indent=2))
        if err:
            raise SystemExit(1)

    extra = {}
    if "output" in ns and "ground_truth" in ns:
        import torch
        output = ns["output"].detach()
        ground_truth = ns["ground_truth"].detach()
        mse_norm = (output - ground_truth).pow(2).mean().item()
        extra["mse_normalized"] = mse_norm
        if "dm" in ns and "ds" in ns:
            dm, ds = ns["dm"], ns["ds"]
            pix_o = torch.clamp(output * ds + dm, 0, 1)
            pix_g = torch.clamp(ground_truth * ds + dm, 0, 1)
            mse_pix = (pix_o - pix_g).pow(2).mean().item()
            extra["mse_after_notebook_denorm"] = mse_pix
            extra["psnr_peak1_after_notebook_denorm"] = (
                float("inf") if mse_pix == 0 else float(10 * torch.log10(torch.tensor(1.0 / mse_pix)))
            )
        (out_dir / "metric_spaces.json").write_text(json.dumps(extra, indent=2))
        print("METRIC_SPACES", json.dumps(extra), flush=True)
    print("DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
