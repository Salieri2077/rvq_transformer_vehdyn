import argparse
import csv
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

try:
    from utils import resolve_default_data_path
except ImportError:
    from rvq_transformer_vehdyn.utils import resolve_default_data_path


def parse_layers_csv(spec: str) -> List[int]:
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer <= 0:
            continue
        layers.append(layer)
    uniq = sorted(set(layers))
    if not uniq:
        raise ValueError("No valid transformer layer value found in --transformer-layers-csv")
    return uniq


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _log(master_log: Path, message: str) -> None:
    line = f"{message}\n"
    print(message, flush=True)
    _append_text(master_log, line)


def run_step(step_name: str, cmd: List[str], log_dir: Path, master_log: Path) -> None:
    step_log = log_dir / f"{step_name}.log"
    _log(master_log, "")
    _log(master_log, f"[{datetime.now().strftime('%F %T')}] STEP={step_name}")
    _log(master_log, f"CMD: {shlex.join(cmd)}")

    with step_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        proc.wait()
        rc = proc.returncode

    if rc != 0:
        _log(master_log, f"[{datetime.now().strftime('%F %T')}] FAILED {step_name}, exit={rc}")
        raise RuntimeError(f"Step failed: {step_name}, exit={rc}")

    _log(master_log, f"[{datetime.now().strftime('%F %T')}] DONE {step_name}")


def main() -> None:
    this_file = Path(__file__).resolve()
    project_root_default = str(this_file.parents[1])
    model_root_default = str(this_file.parents[0] / "work_dirs" / "tokenizer")

    parser = argparse.ArgumentParser(description="Train/eval RVQ TFM with transformer layers sweep.")
    parser.add_argument("--project-root", type=str, default=project_root_default)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--data-path", type=str, default=resolve_default_data_path())
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--transformer-layers-csv", type=str, default="2,3,4,5,6")
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--num-var-plots", type=int, default=1)
    parser.add_argument("--num-worst-plots", type=int, default=1)
    parser.add_argument("--model-root", type=str, default=model_root_default)
    parser.add_argument("--run-root", type=str, default="")
    args = parser.parse_args()

    layers = parse_layers_csv(args.transformer_layers_csv)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root) if args.run_root else Path(args.model_root) / f"multi_tfm_layers_{run_tag}"
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    master_log = log_dir / "master.log"

    project_root = Path(args.project_root)
    train_script = project_root / "rvq_transformer_vehdyn" / "train_tfm.py"
    eval_script = project_root / "rvq_transformer_vehdyn" / "eval_tokenizer_by_scenario.py"

    _log(master_log, f"Run root: {run_root}")
    _log(master_log, f"Transformer layers: {layers}")
    _log(master_log, f"Data path: {args.data_path}")
    _log(master_log, f"Data type: {args.data_type}")
    _log(master_log, f"Master log: {master_log}")

    summary_rows = []
    for tf_layers in layers:
        save_dir = run_root / f"rvq_tfm_kin_tf{tf_layers}"
        save_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            args.python_bin,
            str(train_script),
            "--data-path",
            args.data_path,
            "--save-dir",
            str(save_dir),
            "--data-type",
            args.data_type,
            "--batch-size",
            str(args.batch_size),
            "--num-layers",
            str(args.num_layers),
            "--num-transformer-layers",
            str(tf_layers),
            "--epochs",
            str(args.epochs),
            "--max-samples",
            str(args.max_samples),
        ]
        run_step(f"train_tf{tf_layers}", train_cmd, log_dir, master_log)

        model_path = save_dir / f"{args.data_type}_rvq_taae_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        eval_output_dir = save_dir / "scenario_eval"
        eval_cmd = [
            args.python_bin,
            str(eval_script),
            "--data-path",
            args.data_path,
            "--save-dir",
            str(save_dir),
            "--data-type",
            args.data_type,
            "--batch-size",
            str(args.eval_batch_size),
            "--model-type",
            "taae",
            "--num-var-plots",
            str(args.num_var_plots),
            "--num-worst-plots",
            str(args.num_worst_plots),
            "--num-transformer-layers",
            str(tf_layers),
            "--output-dir",
            str(eval_output_dir),
        ]
        run_step(f"eval_tf{tf_layers}", eval_cmd, log_dir, master_log)

        metrics_csv = eval_output_dir / "scenario_metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing scenario metrics CSV: {metrics_csv}")

        summary_rows.append(
            {
                "transformer_layers": tf_layers,
                "save_dir": str(save_dir),
                "model_path": str(model_path),
                "scenario_eval_dir": str(eval_output_dir),
                "scenario_metrics_csv": str(metrics_csv),
            }
        )

    summary_csv = run_root / "layer_sweep_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transformer_layers",
                "save_dir",
                "model_path",
                "scenario_eval_dir",
                "scenario_metrics_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    _log(master_log, "")
    _log(master_log, "All done.")
    _log(master_log, f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
