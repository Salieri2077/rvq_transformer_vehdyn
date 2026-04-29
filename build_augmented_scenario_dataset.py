import argparse
import os
from typing import Dict, Tuple

import numpy as np

from utils import (
    DEFAULT_AUGMENTED_DATA_PATH,
    DEFAULT_BASE_DATA_PATH,
    build_scenario_masks,
)


def _load_trajs_with_meta(data_path: str) -> Tuple[np.ndarray, Dict]:
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if isinstance(data, dict) and "trajs" in data:
        trajs = np.asarray(data["trajs"])
        meta = {k: v for k, v in data.items() if k != "trajs"}
        return trajs, meta
    return np.asarray(data), {}


def _build_augmented_dataset(
    source_path: str,
    base_path: str,
    output_path: str,
    fps: float,
    scenarios,
):
    src_trajs, _ = _load_trajs_with_meta(source_path)
    base_trajs, base_meta = _load_trajs_with_meta(base_path)

    categories, _ = build_scenario_masks(src_trajs, fps=fps)

    selected_mask = np.zeros(len(src_trajs), dtype=bool)
    scenario_counts = {}
    for scenario in scenarios:
        if scenario not in categories:
            raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(categories.keys())}")
        scenario_mask = categories[scenario]
        selected_mask |= scenario_mask
        scenario_counts[scenario] = int(scenario_mask.sum())

    selected_trajs = src_trajs[selected_mask]
    merged_trajs = np.concatenate([base_trajs, selected_trajs], axis=0)

    out_dict = dict(base_meta)
    out_dict["trajs"] = merged_trajs
    out_dict["mean_val"] = np.mean(merged_trajs, axis=(0, 1), keepdims=True)
    out_dict["std_val"] = np.std(merged_trajs, axis=(0, 1), keepdims=True)
    out_dict["augmented_meta"] = {
        "source_path": source_path,
        "base_path": base_path,
        "scenarios": list(scenarios),
        "scenario_counts": scenario_counts,
        "num_selected_from_source": int(len(selected_trajs)),
        "num_base": int(len(base_trajs)),
        "num_merged": int(len(merged_trajs)),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, out_dict)

    print("=" * 80)
    print("Augmented dataset generated")
    print(f"source_path: {source_path}")
    print(f"base_path:   {base_path}")
    print(f"output_path: {output_path}")
    print("-" * 80)
    print(f"base_count:       {len(base_trajs)}")
    print(f"selected_count:   {len(selected_trajs)}")
    print(f"merged_count:     {len(merged_trajs)}")
    print("selected by scenario:")
    for k, v in scenario_counts.items():
        print(f"  {k}: {v}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific scenarios from source npy and merge into all_datas as a new augmented npy."
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default="/share-global/zhe.du/planner/planNN2/tokenizer/0124_json/sample_trajectorys_by_scenario_update0210.npy",
    )
    parser.add_argument("--base-path", type=str, default=DEFAULT_BASE_DATA_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_AUGMENTED_DATA_PATH)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["Detour", "DirectUTurn", "Reverse", "HighSpeedStraight_120kmh"],
    )
    args = parser.parse_args()

    _build_augmented_dataset(
        source_path=args.source_path,
        base_path=args.base_path,
        output_path=args.output_path,
        fps=args.fps,
        scenarios=args.scenarios,
    )


if __name__ == "__main__":
    main()

# python rvq_transformer_vehdyn/build_augmented_scenario_dataset.py \
#   --source-path /share-global/zhe.du/planner/planNN2/tokenizer/dxdydyaw_trajs_for_tokenizer_208w.npy \
#   --base-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --output-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --fps 5.0 \
#   --scenarios \
#     Stationary \
#     Reverse \
#     DirectUTurn \
#     Detour \
#     LeftTurn \
#     RightTurn \
#     LowSpeedStraight_10kmh \
#     HighSpeedStraight_80kmh \
#     HighSpeedStraight_120kmh
