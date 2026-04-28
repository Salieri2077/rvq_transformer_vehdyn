import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_tfm import TrajRVQTransformer
from train_tfm_accint import AccFirstRVQTokenizer
from utils import (
    build_scenario_masks,
    compute_extended_kinematic_quantities,
    compute_kinematic_profiles,
    denormalize_trajs_torch,
    load_norm_params_torch,
    load_traj_array,
    longest_true_run,
    normalize_trajs_torch,
    resolve_default_data_path,
)


ModelUnion = Union[TrajRVQTransformer, AccFirstRVQTokenizer]

VIOLATION_TYPES = [
    "cusp",
    "lat_acc",
    "acc",
    "jerk",
    "curvature_rate",
    "kinematic_inconsistency",
]

VIOLATION_WEIGHTS = {
    "cusp": 2.0,
    "lat_acc": 1.3,
    "acc": 1.0,
    "jerk": 1.2,
    "curvature_rate": 1.0,
    "kinematic_inconsistency": 1.0,
}


def infer_model_type(model_path: str, explicit_type: str) -> str:
    if explicit_type != "auto":
        return explicit_type
    basename = os.path.basename(model_path).lower()
    if ("accint" in basename) or ("accfirst" in basename):
        return "accint"
    return "taae"


def build_model(model_type: str, input_steps: int, device: torch.device, dt: float) -> ModelUnion:
    if model_type == "accint":
        model = AccFirstRVQTokenizer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=15,
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=2,
            dt=dt,
        ).to(device)
    else:
        model = TrajRVQTransformer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=15,
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=2,
            dt=dt,
        ).to(device)
    return model


def _normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_weights(model: ModelUnion, model_path: str, device: torch.device):
    raw = torch.load(model_path, map_location=device)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint format not supported for: {model_path}")
    state_dict = _normalize_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to strictly load checkpoint={model_path}\n"
            f"missing_keys={missing}\nunexpected_keys={unexpected}"
        )
    model.eval()


def reconstruct_trajs(
    model: ModelUnion,
    trajs: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_recons: List[np.ndarray] = []
    all_codes: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(trajs), batch_size):
            end = min(start + batch_size, len(trajs))
            batch = trajs[start:end]
            x_norm = normalize_trajs_torch(batch, mean, std, scale_factor, clip_limit)
            z = model.encode(x_norm)
            _, _, codes = model.rvq(z)
            x_recon_norm = model.decode_from_codes(codes)
            x_recon = denormalize_trajs_torch(x_recon_norm, mean, std, scale_factor)
            all_recons.append(x_recon.detach().cpu().numpy())
            all_codes.append(codes.detach().cpu().numpy())
    return np.concatenate(all_recons, axis=0), np.concatenate(all_codes, axis=0)


def percentile_to_100_scale(x: float) -> float:
    if x <= 1.0:
        return 100.0 * x
    return x


def safe_percentile(values: np.ndarray, p: float, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, p))


def build_thresholds(
    gt_profile: Dict[str, np.ndarray],
    gt_ext: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    manual = {
        "cusp_turn_angle_rad": args.thr_cusp_turn_angle_rad,
        "cusp_min_speed_mps": args.thr_cusp_min_speed_mps,
        "cusp_curvature_min_1pm": args.thr_cusp_curvature_min_1pm,
        "lat_acc_mps2": args.thr_lat_acc_mps2,
        "acc_mps2": args.thr_acc_mps2,
        "jerk_mps3": args.thr_jerk_mps3,
        "curvature_rate_1pmps": args.thr_curvature_rate_1pmps,
        "kinematic_error_radps": args.thr_kinematic_error_radps,
        "lat_acc_min_speed_mps": args.lat_acc_min_speed_mps,
        "curvature_rate_min_speed_mps": args.curvature_rate_min_speed_mps,
        "kinematic_min_speed_mps": args.kinematic_min_speed_mps,
    }
    if args.threshold_mode == "manual":
        return {
            "mode": "manual",
            "thresholds": manual,
            "meta": {"notes": "manual thresholds (possibly overridden by CLI)."},
        }

    q_main = percentile_to_100_scale(args.threshold_quantile)
    q_cusp = percentile_to_100_scale(args.cusp_quantile)
    q_curv_cusp = percentile_to_100_scale(args.cusp_curvature_quantile)
    margin = float(args.threshold_margin)
    cusp_margin = float(args.cusp_margin)

    speed = gt_profile["speed"]
    lat_mask = speed >= args.lat_acc_min_speed_mps
    curv_rate_mask = speed >= args.curvature_rate_min_speed_mps
    kin_mask = speed >= args.kinematic_min_speed_mps
    cusp_speed_pair_mask = (
        (speed >= args.thr_cusp_min_speed_mps)
        & (gt_ext["speed_prev"] >= args.thr_cusp_min_speed_mps)
    )

    lat_pool = gt_ext["abs_lat_acc"][lat_mask]
    acc_pool = gt_profile["acc"].reshape(-1)
    jerk_pool = gt_ext["jerk_abs"].reshape(-1)
    curv_rate_pool = gt_ext["abs_curvature_rate"][curv_rate_mask]
    kin_pool = gt_ext["kinematic_error"][kin_mask]
    cusp_angle_pool = gt_ext["turn_angle_jump"][cusp_speed_pair_mask]
    cusp_curv_pool = np.abs(gt_ext["curvature_raw"][lat_mask])

    adaptive = dict(manual)
    adaptive["lat_acc_mps2"] = max(manual["lat_acc_mps2"], safe_percentile(lat_pool, q_main, manual["lat_acc_mps2"]) * margin)
    adaptive["acc_mps2"] = max(manual["acc_mps2"], safe_percentile(acc_pool, q_main, manual["acc_mps2"]) * margin)
    adaptive["jerk_mps3"] = max(manual["jerk_mps3"], safe_percentile(jerk_pool, q_main, manual["jerk_mps3"]) * margin)
    adaptive["curvature_rate_1pmps"] = max(
        manual["curvature_rate_1pmps"],
        safe_percentile(curv_rate_pool, q_main, manual["curvature_rate_1pmps"]) * margin,
    )
    adaptive["kinematic_error_radps"] = max(
        manual["kinematic_error_radps"],
        safe_percentile(kin_pool, q_main, manual["kinematic_error_radps"]) * margin,
    )
    adaptive["cusp_turn_angle_rad"] = min(
        np.pi,
        max(
            manual["cusp_turn_angle_rad"],
            safe_percentile(cusp_angle_pool, q_cusp, manual["cusp_turn_angle_rad"]) * cusp_margin,
        ),
    )
    adaptive["cusp_curvature_min_1pm"] = max(
        manual["cusp_curvature_min_1pm"],
        safe_percentile(cusp_curv_pool, q_curv_cusp, manual["cusp_curvature_min_1pm"]) * cusp_margin,
    )

    return {
        "mode": "quantile",
        "thresholds": adaptive,
        "meta": {
            "threshold_quantile": q_main,
            "cusp_quantile": q_cusp,
            "cusp_curvature_quantile": q_curv_cusp,
            "threshold_margin": margin,
            "cusp_margin": cusp_margin,
            "pool_sizes": {
                "lat_pool": int(lat_pool.size),
                "acc_pool": int(acc_pool.size),
                "jerk_pool": int(jerk_pool.size),
                "curvature_rate_pool": int(curv_rate_pool.size),
                "kinematic_pool": int(kin_pool.size),
                "cusp_angle_pool": int(cusp_angle_pool.size),
                "cusp_curvature_pool": int(cusp_curv_pool.size),
            },
        },
    }


def scalar_severity(max_value: float, threshold: float, exceed_ratio: float) -> float:
    if threshold <= 1e-8:
        return float(exceed_ratio)
    exceed_mag = max(0.0, max_value / (threshold + 1e-8) - 1.0)
    return float(exceed_mag + exceed_ratio)


def scenario_labels_from_masks(categories: Dict[str, np.ndarray], n: int) -> Tuple[List[str], List[str], Dict[str, np.ndarray]]:
    if not categories:
        categories = {"All": np.ones(n, dtype=bool)}

    scenario_names = list(categories.keys())
    coverage = np.zeros(n, dtype=bool)
    for name in scenario_names:
        mask = np.asarray(categories[name], dtype=bool)
        if mask.shape[0] != n:
            raise ValueError(f"Scenario mask length mismatch for {name}: expected {n}, got {mask.shape[0]}")
        coverage |= mask

    if not np.all(coverage):
        categories["Other"] = ~coverage
        scenario_names = list(categories.keys())

    labels = []
    for i in range(n):
        found = "Other"
        for name in scenario_names:
            if categories[name][i]:
                found = name
                break
        labels.append(found)

    return labels, scenario_names, categories


def evaluate_physics_for_trajs(
    model_name: str,
    trajs: np.ndarray,
    thresholds: Dict[str, float],
    scenario_labels: List[str],
    dt: float,
    hard_multiplier: float,
    hard_min_consecutive: int,
    force_no_hard_impossible: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, List[int]]]]:
    profile = compute_kinematic_profiles(trajs, dt=dt)
    ext = compute_extended_kinematic_quantities(
        profile,
        dt=dt,
        curvature_clip=2.0,
        jerk_smooth_window=3,
    )

    n, t = profile["speed"].shape
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, List[int]]] = []

    for i in range(n):
        speed = profile["speed"][i]
        acc = profile["acc"][i]
        curvature = ext["curvature_raw"][i]
        lat_acc_abs = ext["abs_lat_acc"][i]
        jerk_abs = ext["jerk_abs"][i]
        curv_rate_abs = ext["abs_curvature_rate"][i]
        kin_err = ext["kinematic_error"][i]
        turn_angle_jump = ext["turn_angle_jump"][i]
        turn_dot = ext["turn_dot"][i]
        speed_prev = ext["speed_prev"][i]

        cusp_speed_mask = (
            (speed >= thresholds["cusp_min_speed_mps"])
            & (speed_prev >= thresholds["cusp_min_speed_mps"])
        )
        cusp_angle_mask = turn_angle_jump >= thresholds["cusp_turn_angle_rad"]
        cusp_curv_mask = np.abs(curvature) >= thresholds["cusp_curvature_min_1pm"]
        cusp_reverse_mask = turn_dot <= -0.2
        cusp_events = cusp_speed_mask & cusp_angle_mask & (cusp_curv_mask | cusp_reverse_mask)

        max_turn_angle_jump = float(np.max(turn_angle_jump)) if t > 0 else 0.0
        cusp_count = int(np.sum(cusp_events))
        cusp_flag = cusp_count > 0
        if cusp_flag:
            cusp_severity = scalar_severity(
                max_value=max_turn_angle_jump,
                threshold=thresholds["cusp_turn_angle_rad"],
                exceed_ratio=float(cusp_count / max(t, 1)),
            )
        else:
            cusp_severity = 0.0

        lat_valid = speed >= thresholds["lat_acc_min_speed_mps"]
        lat_exceed = lat_valid & (lat_acc_abs > thresholds["lat_acc_mps2"])
        lat_valid_count = int(np.sum(lat_valid))
        lat_exceed_ratio = float(np.sum(lat_exceed) / max(lat_valid_count, 1))
        max_abs_lat_acc = float(np.max(lat_acc_abs[lat_valid])) if lat_valid_count > 0 else 0.0
        lat_acc_flag = lat_exceed_ratio > 0.0
        lat_acc_severity = scalar_severity(max_abs_lat_acc, thresholds["lat_acc_mps2"], lat_exceed_ratio)

        acc_abs = np.abs(acc)
        acc_exceed = acc_abs > thresholds["acc_mps2"]
        acc_exceed_ratio = float(np.mean(acc_exceed))
        max_abs_acc = float(np.max(acc_abs)) if t > 0 else 0.0
        acc_flag = acc_exceed_ratio > 0.0
        acc_severity = scalar_severity(max_abs_acc, thresholds["acc_mps2"], acc_exceed_ratio)

        jerk_exceed = jerk_abs > thresholds["jerk_mps3"]
        jerk_exceed_ratio = float(np.mean(jerk_exceed))
        max_abs_jerk = float(np.max(jerk_abs)) if t > 0 else 0.0
        jerk_flag = jerk_exceed_ratio > 0.0
        jerk_severity = scalar_severity(max_abs_jerk, thresholds["jerk_mps3"], jerk_exceed_ratio)

        curv_rate_valid = speed >= thresholds["curvature_rate_min_speed_mps"]
        curv_rate_exceed = curv_rate_valid & (curv_rate_abs > thresholds["curvature_rate_1pmps"])
        curv_rate_valid_count = int(np.sum(curv_rate_valid))
        curv_rate_exceed_ratio = float(np.sum(curv_rate_exceed) / max(curv_rate_valid_count, 1))
        max_abs_curvature_rate = (
            float(np.max(curv_rate_abs[curv_rate_valid])) if curv_rate_valid_count > 0 else 0.0
        )
        curv_rate_flag = curv_rate_exceed_ratio > 0.0
        curv_rate_severity = scalar_severity(
            max_abs_curvature_rate,
            thresholds["curvature_rate_1pmps"],
            curv_rate_exceed_ratio,
        )

        kin_valid = speed >= thresholds["kinematic_min_speed_mps"]
        kin_exceed = kin_valid & (kin_err > thresholds["kinematic_error_radps"])
        kin_valid_count = int(np.sum(kin_valid))
        kin_exceed_ratio = float(np.sum(kin_exceed) / max(kin_valid_count, 1))
        mean_kin_err = float(np.mean(kin_err[kin_valid])) if kin_valid_count > 0 else 0.0
        max_kin_err = float(np.max(kin_err[kin_valid])) if kin_valid_count > 0 else 0.0
        kin_flag = kin_exceed_ratio > 0.0
        kin_severity = scalar_severity(max_kin_err, thresholds["kinematic_error_radps"], kin_exceed_ratio)

        num_violation_types = int(
            cusp_flag + lat_acc_flag + acc_flag + jerk_flag + curv_rate_flag + kin_flag
        )
        hard_impossible_flag = bool(
            cusp_flag
            or (
                max_abs_lat_acc > thresholds["lat_acc_mps2"] * hard_multiplier
                and longest_true_run(lat_exceed) >= hard_min_consecutive
            )
            or (
                max_abs_acc > thresholds["acc_mps2"] * hard_multiplier
                and longest_true_run(acc_exceed) >= hard_min_consecutive
            )
            or (
                max_abs_jerk > thresholds["jerk_mps3"] * hard_multiplier
                and longest_true_run(jerk_exceed) >= hard_min_consecutive
            )
        )
        if force_no_hard_impossible:
            hard_impossible_flag = False

        physics_violation_score = float(
            VIOLATION_WEIGHTS["cusp"] * cusp_severity
            + VIOLATION_WEIGHTS["lat_acc"] * lat_acc_severity
            + VIOLATION_WEIGHTS["acc"] * acc_severity
            + VIOLATION_WEIGHTS["jerk"] * jerk_severity
            + VIOLATION_WEIGHTS["curvature_rate"] * curv_rate_severity
            + VIOLATION_WEIGHTS["kinematic_inconsistency"] * kin_severity
        )
        any_violation_flag = num_violation_types > 0

        row = {
            "model": model_name,
            "sample_idx": i,
            "scenario": scenario_labels[i],
            "cusp_flag": int(cusp_flag),
            "cusp_count": cusp_count,
            "max_turn_angle_jump": max_turn_angle_jump,
            "cusp_severity": cusp_severity,
            "lat_acc_violation_flag": int(lat_acc_flag),
            "max_abs_lat_acc": max_abs_lat_acc,
            "lat_acc_exceed_ratio": lat_exceed_ratio,
            "lat_acc_severity": lat_acc_severity,
            "acc_violation_flag": int(acc_flag),
            "max_abs_acc": max_abs_acc,
            "acc_exceed_ratio": acc_exceed_ratio,
            "acc_severity": acc_severity,
            "jerk_violation_flag": int(jerk_flag),
            "max_abs_jerk": max_abs_jerk,
            "jerk_exceed_ratio": jerk_exceed_ratio,
            "jerk_severity": jerk_severity,
            "curvature_rate_violation_flag": int(curv_rate_flag),
            "max_abs_curvature_rate": max_abs_curvature_rate,
            "curvature_rate_exceed_ratio": curv_rate_exceed_ratio,
            "curvature_rate_severity": curv_rate_severity,
            "kinematic_inconsistency_flag": int(kin_flag),
            "mean_kinematic_error": mean_kin_err,
            "max_kinematic_error": max_kin_err,
            "kinematic_inconsistency_severity": kin_severity,
            "any_physics_violation_flag": int(any_violation_flag),
            "num_violation_types": num_violation_types,
            "physics_violation_score": physics_violation_score,
            "hard_impossible_flag": int(hard_impossible_flag),
        }
        rows.append(row)

        sample_events = {
            "cusp": np.where(cusp_events)[0].astype(int).tolist(),
            "lat_acc": np.where(lat_exceed)[0].astype(int).tolist(),
            "acc": np.where(acc_exceed)[0].astype(int).tolist(),
            "jerk": np.where(jerk_exceed)[0].astype(int).tolist(),
            "curvature_rate": np.where(curv_rate_exceed)[0].astype(int).tolist(),
            "kinematic_inconsistency": np.where(kin_exceed)[0].astype(int).tolist(),
        }
        events.append(sample_events)

    return rows, events


def summarize_overall(rows: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"model": model_name, "count": 0}

    out: Dict[str, Any] = {
        "model": model_name,
        "count": n,
        "impossible_ratio": float(np.mean([r["hard_impossible_flag"] for r in rows])),
        "any_violation_ratio": float(np.mean([r["any_physics_violation_flag"] for r in rows])),
        "avg_physics_violation_score": float(np.mean([r["physics_violation_score"] for r in rows])),
        "mean_num_violation_types": float(np.mean([r["num_violation_types"] for r in rows])),
    }
    out["cusp_trigger_ratio"] = float(np.mean([r["cusp_flag"] for r in rows]))
    out["lat_acc_trigger_ratio"] = float(np.mean([r["lat_acc_violation_flag"] for r in rows]))
    out["acc_trigger_ratio"] = float(np.mean([r["acc_violation_flag"] for r in rows]))
    out["jerk_trigger_ratio"] = float(np.mean([r["jerk_violation_flag"] for r in rows]))
    out["curvature_rate_trigger_ratio"] = float(np.mean([r["curvature_rate_violation_flag"] for r in rows]))
    out["kinematic_inconsistency_trigger_ratio"] = float(
        np.mean([r["kinematic_inconsistency_flag"] for r in rows])
    )
    return out


def summarize_per_scenario(rows: List[Dict[str, Any]], scenario_names: List[str], model_name: str) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for scenario in scenario_names:
        sub = [r for r in rows if r["scenario"] == scenario]
        if len(sub) == 0:
            out_rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "count": 0,
                    "impossible_ratio": 0.0,
                    "any_violation_ratio": 0.0,
                    "avg_physics_violation_score": 0.0,
                    "mean_num_violation_types": 0.0,
                    "top_violation_types": "",
                }
            )
            continue

        v_ratios = {
            "cusp": float(np.mean([x["cusp_flag"] for x in sub])),
            "lat_acc": float(np.mean([x["lat_acc_violation_flag"] for x in sub])),
            "acc": float(np.mean([x["acc_violation_flag"] for x in sub])),
            "jerk": float(np.mean([x["jerk_violation_flag"] for x in sub])),
            "curvature_rate": float(np.mean([x["curvature_rate_violation_flag"] for x in sub])),
            "kinematic_inconsistency": float(np.mean([x["kinematic_inconsistency_flag"] for x in sub])),
        }
        top_types = sorted(v_ratios.items(), key=lambda kv: kv[1], reverse=True)
        top_types = [name for name, value in top_types if value > 0.0][:3]

        row = {
            "model": model_name,
            "scenario": scenario,
            "count": len(sub),
            "impossible_ratio": float(np.mean([x["hard_impossible_flag"] for x in sub])),
            "any_violation_ratio": float(np.mean([x["any_physics_violation_flag"] for x in sub])),
            "avg_physics_violation_score": float(np.mean([x["physics_violation_score"] for x in sub])),
            "mean_num_violation_types": float(np.mean([x["num_violation_types"] for x in sub])),
            "top_violation_types": "|".join(top_types),
            "cusp_ratio": v_ratios["cusp"],
            "lat_acc_ratio": v_ratios["lat_acc"],
            "acc_ratio": v_ratios["acc"],
            "jerk_ratio": v_ratios["jerk"],
            "curvature_rate_ratio": v_ratios["curvature_rate"],
            "kinematic_inconsistency_ratio": v_ratios["kinematic_inconsistency"],
        }
        out_rows.append(row)
    return out_rows


def summarize_per_violation(rows: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    mapping = {
        "cusp": ("cusp_flag", "cusp_severity", "cusp_count", "max_turn_angle_jump"),
        "lat_acc": ("lat_acc_violation_flag", "lat_acc_severity", "lat_acc_exceed_ratio", "max_abs_lat_acc"),
        "acc": ("acc_violation_flag", "acc_severity", "acc_exceed_ratio", "max_abs_acc"),
        "jerk": ("jerk_violation_flag", "jerk_severity", "jerk_exceed_ratio", "max_abs_jerk"),
        "curvature_rate": (
            "curvature_rate_violation_flag",
            "curvature_rate_severity",
            "curvature_rate_exceed_ratio",
            "max_abs_curvature_rate",
        ),
        "kinematic_inconsistency": (
            "kinematic_inconsistency_flag",
            "kinematic_inconsistency_severity",
            "mean_kinematic_error",
            "max_kinematic_error",
        ),
    }
    out_rows: List[Dict[str, Any]] = []
    for vt, (flag_col, sev_col, ratio_col, max_col) in mapping.items():
        flags = np.array([r[flag_col] for r in rows], dtype=np.float32)
        sevs = np.array([r[sev_col] for r in rows], dtype=np.float32)
        ratios = np.array([r[ratio_col] for r in rows], dtype=np.float32)
        max_vals = np.array([r[max_col] for r in rows], dtype=np.float32)
        out_rows.append(
            {
                "model": model_name,
                "violation_type": vt,
                "trigger_ratio": float(np.mean(flags)),
                "mean_severity": float(np.mean(sevs)),
                "mean_ratio_or_count": float(np.mean(ratios)),
                "mean_max_value": float(np.mean(max_vals)),
                "p95_max_value": float(np.percentile(max_vals, 95)),
            }
        )
    return out_rows


def build_scenario_violation_matrix(per_scenario_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in per_scenario_rows:
        out.append(
            {
                "model": r["model"],
                "scenario": r["scenario"],
                "cusp": r.get("cusp_ratio", 0.0),
                "lat_acc": r.get("lat_acc_ratio", 0.0),
                "acc": r.get("acc_ratio", 0.0),
                "jerk": r.get("jerk_ratio", 0.0),
                "curvature_rate": r.get("curvature_rate_ratio", 0.0),
                "kinematic_inconsistency": r.get("kinematic_inconsistency_ratio", 0.0),
            }
        )
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_violation_markers(ax, t: np.ndarray, event_indices: List[int], color: str = "red"):
    if not event_indices:
        return
    for idx in event_indices[:100]:
        if 0 <= idx < len(t):
            ax.axvline(t[idx], color=color, linestyle="--", linewidth=0.8, alpha=0.35)


def _format_case_title_fields(violation_type: str, row: Dict[str, Any]) -> str:
    if violation_type == "cusp":
        return f"max_turn={row['max_turn_angle_jump']:.2f}rad, count={row['cusp_count']}"
    if violation_type == "lat_acc":
        return f"max|a_lat|={row['max_abs_lat_acc']:.2f}, ratio={row['lat_acc_exceed_ratio']:.2f}"
    if violation_type == "acc":
        return f"max|a|={row['max_abs_acc']:.2f}, ratio={row['acc_exceed_ratio']:.2f}"
    if violation_type == "jerk":
        return f"max|j|={row['max_abs_jerk']:.2f}, ratio={row['jerk_exceed_ratio']:.2f}"
    if violation_type == "curvature_rate":
        return (
            f"max|dkappa/dt|={row['max_abs_curvature_rate']:.2f}, "
            f"ratio={row['curvature_rate_exceed_ratio']:.2f}"
        )
    if violation_type == "kinematic_inconsistency":
        return (
            f"mean_kin={row['mean_kinematic_error']:.2f}, "
            f"max_kin={row['max_kinematic_error']:.2f}"
        )
    return ""


def plot_case(
    model_name: str,
    sample_idx: int,
    scenario_name: str,
    violation_type: str,
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    row: Dict[str, Any],
    event_indices: List[int],
    dt: float,
    save_path: str,
    dpi: int = 180,
):
    gt_prof = compute_kinematic_profiles(gt_traj[None, ...], dt=dt)
    pred_prof = compute_kinematic_profiles(pred_traj[None, ...], dt=dt)

    gt_xy = gt_prof["xy"][0]
    pred_xy = pred_prof["xy"][0]
    t = np.arange(gt_traj.shape[0]) * dt

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ax_traj = axes[0, 0]
    ax_vx = axes[0, 1]
    ax_vy = axes[0, 2]
    ax_v = axes[0, 3]
    ax_curv = axes[1, 0]
    ax_ax = axes[1, 1]
    ax_ay = axes[1, 2]
    ax_a = axes[1, 3]

    ax_traj.plot(gt_xy[:, 0], gt_xy[:, 1], label="GT", linewidth=2.2)
    ax_traj.plot(pred_xy[:, 0], pred_xy[:, 1], label="Recon", linewidth=2.0, linestyle="--")
    ax_traj.scatter(gt_xy[0, 0], gt_xy[0, 1], c="green", s=45, label="Start")
    ax_traj.scatter(gt_xy[-1, 0], gt_xy[-1, 1], c="red", s=45, label="GT End")
    ax_traj.scatter(pred_xy[-1, 0], pred_xy[-1, 1], c="orange", s=45, label="Recon End")
    if event_indices:
        event_xy = pred_xy[np.array(event_indices, dtype=np.int64)]
        ax_traj.scatter(event_xy[:, 0], event_xy[:, 1], c="red", s=50, marker="x", label="Violation")
    ax_traj.set_title("Trajectory (global XY)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=8)

    ax_vx.plot(t, gt_prof["vx"][0], label="GT vx", linewidth=1.8)
    ax_vx.plot(t, pred_prof["vx"][0], label="Recon vx", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_vx, t, event_indices)
    ax_vx.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_vx.set_title("vx")
    ax_vx.set_xlabel("Time (s)")
    ax_vx.set_ylabel("m/s")
    ax_vx.grid(True, alpha=0.3)
    ax_vx.legend(fontsize=8)

    ax_vy.plot(t, gt_prof["vy"][0], label="GT vy", linewidth=1.8)
    ax_vy.plot(t, pred_prof["vy"][0], label="Recon vy", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_vy, t, event_indices)
    ax_vy.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_vy.set_title("vy")
    ax_vy.set_xlabel("Time (s)")
    ax_vy.set_ylabel("m/s")
    ax_vy.grid(True, alpha=0.3)
    ax_vy.legend(fontsize=8)

    ax_v.plot(t, gt_prof["speed"][0], label="GT |v|", linewidth=1.8)
    ax_v.plot(t, pred_prof["speed"][0], label="Recon |v|", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_v, t, event_indices)
    ax_v.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_v.set_title("Speed |v|")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("m/s")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(fontsize=8)

    ax_curv.plot(t, gt_prof["curvature"][0], label="GT curvature", linewidth=1.8)
    ax_curv.plot(t, pred_prof["curvature"][0], label="Recon curvature", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_curv, t, event_indices)
    ax_curv.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_curv.set_title("Curvature")
    ax_curv.set_xlabel("Time (s)")
    ax_curv.set_ylabel("1/m")
    ax_curv.grid(True, alpha=0.3)
    ax_curv.legend(fontsize=8)

    ax_ax.plot(t, gt_prof["ax"][0], label="GT ax", linewidth=1.8)
    ax_ax.plot(t, pred_prof["ax"][0], label="Recon ax", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_ax, t, event_indices)
    ax_ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_ax.set_title("ax")
    ax_ax.set_xlabel("Time (s)")
    ax_ax.set_ylabel("m/s^2")
    ax_ax.grid(True, alpha=0.3)
    ax_ax.legend(fontsize=8)

    ax_ay.plot(t, gt_prof["ay"][0], label="GT ay", linewidth=1.8)
    ax_ay.plot(t, pred_prof["ay"][0], label="Recon ay", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_ay, t, event_indices)
    ax_ay.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_ay.set_title("ay")
    ax_ay.set_xlabel("Time (s)")
    ax_ay.set_ylabel("m/s^2")
    ax_ay.grid(True, alpha=0.3)
    ax_ay.legend(fontsize=8)

    ax_a.plot(t, gt_prof["acc"][0], label="GT |a|", linewidth=1.8)
    ax_a.plot(t, pred_prof["acc"][0], label="Recon |a|", linewidth=1.8, linestyle="--")
    _add_violation_markers(ax_a, t, event_indices)
    ax_a.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_a.set_title("|a|")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("m/s^2")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=8)

    key_text = _format_case_title_fields(violation_type, row)
    fig.suptitle(
        f"model={model_name} | idx={sample_idx} | scenario={scenario_name} | "
        f"violation={violation_type} | score={row['physics_violation_score']:.3f}\n{key_text}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_topk_plots(
    model_name: str,
    gt_trajs: np.ndarray,
    pred_trajs: np.ndarray,
    rows: List[Dict[str, Any]],
    events: List[Dict[str, List[int]]],
    scenario_names: List[str],
    top_k: int,
    dt: float,
    save_dir: str,
    dpi: int,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []

    flag_col = {
        "cusp": "cusp_flag",
        "lat_acc": "lat_acc_violation_flag",
        "acc": "acc_violation_flag",
        "jerk": "jerk_violation_flag",
        "curvature_rate": "curvature_rate_violation_flag",
        "kinematic_inconsistency": "kinematic_inconsistency_flag",
    }
    sev_col = {
        "cusp": "cusp_severity",
        "lat_acc": "lat_acc_severity",
        "acc": "acc_severity",
        "jerk": "jerk_severity",
        "curvature_rate": "curvature_rate_severity",
        "kinematic_inconsistency": "kinematic_inconsistency_severity",
    }

    case_rows: List[Dict[str, Any]] = []
    for scenario_name in scenario_names:
        scenario_indices = [i for i, r in enumerate(rows) if r["scenario"] == scenario_name]
        for vt in VIOLATION_TYPES:
            candidates = [
                i for i in scenario_indices if int(rows[i][flag_col[vt]]) == 1
            ]
            if not candidates:
                continue
            candidates = sorted(candidates, key=lambda idx: rows[idx][sev_col[vt]], reverse=True)[:top_k]
            for rank, idx in enumerate(candidates, start=1):
                plot_path = os.path.join(
                    save_dir,
                    "plots",
                    model_name,
                    scenario_name,
                    vt,
                    f"case_{rank:02d}_idx_{idx:06d}.png",
                )
                plot_case(
                    model_name=model_name,
                    sample_idx=idx,
                    scenario_name=scenario_name,
                    violation_type=vt,
                    gt_traj=gt_trajs[idx],
                    pred_traj=pred_trajs[idx],
                    row=rows[idx],
                    event_indices=events[idx].get(vt, []),
                    dt=dt,
                    save_path=plot_path,
                    dpi=dpi,
                )
                case_rows.append(
                    {
                        "model": model_name,
                        "scenario": scenario_name,
                        "violation_type": vt,
                        "rank": rank,
                        "sample_idx": idx,
                        "severity": rows[idx][sev_col[vt]],
                        "plot_path": plot_path,
                    }
                )
    return case_rows


def _bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    save_path: str,
    color: str = "#4C72B0",
):
    if not labels:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _grouped_bar(
    categories: List[str],
    model_names: List[str],
    values_by_model: Dict[str, List[float]],
    title: str,
    ylabel: str,
    save_path: str,
):
    if not categories or not model_names:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 0.8), 5))
    x = np.arange(len(categories))
    width = 0.8 / max(len(model_names), 1)
    for i, model_name in enumerate(model_names):
        vals = values_by_model.get(model_name, [0.0] * len(categories))
        ax.bar(x + (i - (len(model_names) - 1) / 2.0) * width, vals, width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _heatmap(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    save_path: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(x_labels) * 1.0), max(5, len(y_labels) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0.0, vmax=max(1e-6, np.max(matrix)))
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Trigger Ratio")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_summary_figures(
    overall_rows: List[Dict[str, Any]],
    per_scenario_rows: List[Dict[str, Any]],
    per_violation_rows: List[Dict[str, Any]],
    scenario_names: List[str],
    save_dir: str,
):
    fig_dir = os.path.join(save_dir, "figures")
    model_names = [r["model"] for r in overall_rows]

    _bar_plot(
        labels=model_names,
        values=[r.get("impossible_ratio", 0.0) for r in overall_rows],
        title="Overall Impossible Ratio",
        ylabel="Ratio",
        save_path=os.path.join(fig_dir, "overall_impossible_ratio.png"),
    )
    _bar_plot(
        labels=model_names,
        values=[r.get("avg_physics_violation_score", 0.0) for r in overall_rows],
        title="Overall Physics Violation Score",
        ylabel="Score",
        save_path=os.path.join(fig_dir, "overall_physics_violation_score.png"),
        color="#DD8452",
    )

    values_by_model_impossible: Dict[str, List[float]] = {}
    for model in model_names:
        rows = [r for r in per_scenario_rows if r["model"] == model]
        rows_by_scenario = {r["scenario"]: r for r in rows}
        values_by_model_impossible[model] = [
            rows_by_scenario.get(s, {}).get("impossible_ratio", 0.0) for s in scenario_names
        ]
    _grouped_bar(
        categories=scenario_names,
        model_names=model_names,
        values_by_model=values_by_model_impossible,
        title="Scenario-wise Impossible Ratio",
        ylabel="Ratio",
        save_path=os.path.join(fig_dir, "scenario_impossible_ratio_comparison.png"),
    )

    violation_names = list(VIOLATION_TYPES)
    values_by_model_violation: Dict[str, List[float]] = {}
    for model in model_names:
        rows = [r for r in per_violation_rows if r["model"] == model]
        rows_by_v = {r["violation_type"]: r for r in rows}
        values_by_model_violation[model] = [
            rows_by_v.get(v, {}).get("trigger_ratio", 0.0) for v in violation_names
        ]
    _grouped_bar(
        categories=violation_names,
        model_names=model_names,
        values_by_model=values_by_model_violation,
        title="Violation Trigger Ratio by Type",
        ylabel="Ratio",
        save_path=os.path.join(fig_dir, "violation_trigger_ratio_comparison.png"),
    )

    for model in model_names:
        rows = [r for r in per_scenario_rows if r["model"] == model]
        rows_by_s = {r["scenario"]: r for r in rows}
        mat = np.zeros((len(scenario_names), len(violation_names)), dtype=np.float32)
        for i, s in enumerate(scenario_names):
            rr = rows_by_s.get(s, {})
            mat[i, :] = [
                rr.get("cusp_ratio", 0.0),
                rr.get("lat_acc_ratio", 0.0),
                rr.get("acc_ratio", 0.0),
                rr.get("jerk_ratio", 0.0),
                rr.get("curvature_rate_ratio", 0.0),
                rr.get("kinematic_inconsistency_ratio", 0.0),
            ]
        _heatmap(
            matrix=mat,
            x_labels=violation_names,
            y_labels=scenario_names,
            title=f"Scenario x Violation Heatmap ({model})",
            save_path=os.path.join(fig_dir, f"scenario_violation_heatmap_{model}.png"),
        )


def resolve_norm_path(
    explicit_path: Optional[str],
    shared_norm_path: Optional[str],
    model_path: str,
    data_type: str,
) -> str:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if shared_norm_path:
        candidates.append(shared_norm_path)
    candidates.append(os.path.join(os.path.dirname(model_path), f"{data_type}_norm_params.pkl"))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Failed to resolve norm params for model={model_path}. Tried: {candidates}"
    )


def print_overall_brief(overall_rows: List[Dict[str, Any]]):
    print("=" * 96)
    print("Overall Physical Feasibility Summary")
    print("=" * 96)
    for r in overall_rows:
        print(
            f"[{r['model']}] impossible_ratio={r.get('impossible_ratio', 0.0):.4f}, "
            f"avg_score={r.get('avg_physics_violation_score', 0.0):.4f}, "
            f"any_violation_ratio={r.get('any_violation_ratio', 0.0):.4f}, "
            f"cusp={r.get('cusp_trigger_ratio', 0.0):.4f}, "
            f"lat_acc={r.get('lat_acc_trigger_ratio', 0.0):.4f}, "
            f"jerk={r.get('jerk_trigger_ratio', 0.0):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate physical feasibility of trajectory reconstructions for two RVQ models.",
    )
    parser.add_argument("--data-path", type=str, default=resolve_default_data_path())
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="Use first N samples for quick evaluation. 0 means full dataset.")

    parser.add_argument("--model1-path", type=str, required=True)
    parser.add_argument("--model1-type", type=str, default="auto", choices=["auto", "taae", "accint"])
    parser.add_argument("--model1-name", type=str, default="model1")
    parser.add_argument("--model1-norm-path", type=str, default=None)

    parser.add_argument("--model2-path", type=str, required=True)
    parser.add_argument("--model2-type", type=str, default="auto", choices=["auto", "taae", "accint"])
    parser.add_argument("--model2-name", type=str, default="model2")
    parser.add_argument("--model2-norm-path", type=str, default=None)

    parser.add_argument("--norm-params-path", type=str, default=None, help="Optional shared norm params path.")
    parser.add_argument("--output-dir", type=str, default="./work_dirs/tokenizer/physics_feasibility_eval")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--include-gt-baseline", action="store_true")

    parser.add_argument("--threshold-mode", type=str, default="quantile", choices=["quantile", "manual"])
    parser.add_argument("--threshold-quantile", type=float, default=99.5)
    parser.add_argument("--threshold-margin", type=float, default=1.1)
    parser.add_argument("--cusp-quantile", type=float, default=99.9)
    parser.add_argument("--cusp-curvature-quantile", type=float, default=97.0)
    parser.add_argument("--cusp-margin", type=float, default=1.0)

    parser.add_argument("--thr-cusp-turn-angle-rad", type=float, default=2.6)
    parser.add_argument("--thr-cusp-min-speed-mps", type=float, default=1.0)
    parser.add_argument("--thr-cusp-curvature-min-1pm", type=float, default=0.06)
    parser.add_argument("--thr-lat-acc-mps2", type=float, default=8.0)
    parser.add_argument("--thr-acc-mps2", type=float, default=10.0)
    parser.add_argument("--thr-jerk-mps3", type=float, default=25.0)
    parser.add_argument("--thr-curvature-rate-1pmps", type=float, default=2.0)
    parser.add_argument("--thr-kinematic-error-radps", type=float, default=1.0)
    parser.add_argument("--lat-acc-min-speed-mps", type=float, default=1.0)
    parser.add_argument("--curvature-rate-min-speed-mps", type=float, default=0.5)
    parser.add_argument("--kinematic-min-speed-mps", type=float, default=0.5)
    parser.add_argument("--hard-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--hard-min-consecutive",
        type=int,
        default=2,
        help="Minimum consecutive violating steps for hard impossible (except cusp).",
    )
    parser.add_argument(
        "--force-gt-feasible",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, GT baseline will not be marked as hard impossible (keeps violation metrics).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for path in [args.data_path, args.model1_path, args.model2_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs = load_traj_array(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    if args.max_samples > 0:
        trajs = trajs[: args.max_samples]
    n_samples, input_steps, _ = trajs.shape

    try:
        categories, _ = build_scenario_masks(trajs, fps=args.fps)
    except Exception as exc:
        print(f"[WARN] build_scenario_masks failed, fallback to single scenario. reason={exc}")
        categories = {"All": np.ones(n_samples, dtype=bool)}
    scenario_labels, scenario_names, categories = scenario_labels_from_masks(categories, n_samples)

    model_cfgs = [
        {
            "name": args.model1_name,
            "path": args.model1_path,
            "type": infer_model_type(args.model1_path, args.model1_type),
            "norm_path": resolve_norm_path(
                args.model1_norm_path,
                args.norm_params_path,
                args.model1_path,
                args.data_type,
            ),
        },
        {
            "name": args.model2_name,
            "path": args.model2_path,
            "type": infer_model_type(args.model2_path, args.model2_type),
            "norm_path": resolve_norm_path(
                args.model2_norm_path,
                args.norm_params_path,
                args.model2_path,
                args.data_type,
            ),
        },
    ]

    reconstructions: Dict[str, np.ndarray] = {}
    code_shapes: Dict[str, List[int]] = {}

    for cfg in model_cfgs:
        print(f"[INFO] Loading model={cfg['name']} type={cfg['type']} checkpoint={cfg['path']}")
        model = build_model(model_type=cfg["type"], input_steps=input_steps, device=device, dt=args.dt)
        load_model_weights(model, cfg["path"], device=device)
        norm = load_norm_params_torch(cfg["norm_path"], device=device)
        model.set_norm_params(norm["mean"], norm["std"], norm["scale_factor"])
        recon, codes = reconstruct_trajs(
            model=model,
            trajs=trajs,
            mean=norm["mean"],
            std=norm["std"],
            scale_factor=norm["scale_factor"],
            clip_limit=norm["clip_limit"],
            batch_size=args.batch_size,
        )
        reconstructions[cfg["name"]] = recon.astype(np.float32)
        code_shapes[cfg["name"]] = list(codes.shape)

    gt_profile = compute_kinematic_profiles(trajs, dt=args.dt)
    gt_ext = compute_extended_kinematic_quantities(
        gt_profile,
        dt=args.dt,
        curvature_clip=2.0,
        jerk_smooth_window=3,
    )
    threshold_obj = build_thresholds(gt_profile=gt_profile, gt_ext=gt_ext, args=args)
    thresholds = threshold_obj["thresholds"]

    overall_rows: List[Dict[str, Any]] = []
    per_scenario_rows: List[Dict[str, Any]] = []
    per_violation_rows: List[Dict[str, Any]] = []
    scenario_matrix_rows: List[Dict[str, Any]] = []
    topk_case_rows: List[Dict[str, Any]] = []
    all_per_traj_rows: List[Dict[str, Any]] = []

    model_rows_map: Dict[str, List[Dict[str, Any]]] = {}

    for cfg in model_cfgs:
        model_name = cfg["name"]
        pred_trajs = reconstructions[model_name]
        rows, events = evaluate_physics_for_trajs(
            model_name=model_name,
            trajs=pred_trajs,
            thresholds=thresholds,
            scenario_labels=scenario_labels,
            dt=args.dt,
            hard_multiplier=args.hard_multiplier,
            hard_min_consecutive=args.hard_min_consecutive,
            force_no_hard_impossible=False,
        )
        model_rows_map[model_name] = rows
        all_per_traj_rows.extend(rows)

        overall = summarize_overall(rows, model_name=model_name)
        overall_rows.append(overall)

        scenario_rows = summarize_per_scenario(rows, scenario_names=scenario_names, model_name=model_name)
        per_scenario_rows.extend(scenario_rows)

        violation_rows = summarize_per_violation(rows, model_name=model_name)
        per_violation_rows.extend(violation_rows)

        scenario_matrix_rows.extend(build_scenario_violation_matrix(scenario_rows))

        topk_case_rows.extend(
            generate_topk_plots(
                model_name=model_name,
                gt_trajs=trajs,
                pred_trajs=pred_trajs,
                rows=rows,
                events=events,
                scenario_names=scenario_names,
                top_k=args.top_k,
                dt=args.dt,
                save_dir=args.output_dir,
                dpi=args.plot_dpi,
            )
        )

    if args.include_gt_baseline:
        gt_rows, gt_events = evaluate_physics_for_trajs(
            model_name="GT",
            trajs=trajs,
            thresholds=thresholds,
            scenario_labels=scenario_labels,
            dt=args.dt,
            hard_multiplier=args.hard_multiplier,
            hard_min_consecutive=args.hard_min_consecutive,
            force_no_hard_impossible=bool(args.force_gt_feasible),
        )
        model_rows_map["GT"] = gt_rows
        all_per_traj_rows.extend(gt_rows)
        overall_rows.append(summarize_overall(gt_rows, model_name="GT"))
        gt_scenario_rows = summarize_per_scenario(gt_rows, scenario_names=scenario_names, model_name="GT")
        per_scenario_rows.extend(gt_scenario_rows)
        per_violation_rows.extend(summarize_per_violation(gt_rows, model_name="GT"))
        scenario_matrix_rows.extend(build_scenario_violation_matrix(gt_scenario_rows))

        if args.top_k > 0:
            topk_case_rows.extend(
                generate_topk_plots(
                    model_name="GT",
                    gt_trajs=trajs,
                    pred_trajs=trajs,
                    rows=gt_rows,
                    events=gt_events,
                    scenario_names=scenario_names,
                    top_k=min(args.top_k, 1),
                    dt=args.dt,
                    save_dir=args.output_dir,
                    dpi=args.plot_dpi,
                )
            )

    ranked = sorted(
        overall_rows,
        key=lambda r: (r.get("avg_physics_violation_score", 1e9), r.get("impossible_ratio", 1e9)),
    )
    rank_map = {r["model"]: i + 1 for i, r in enumerate(ranked)}
    for r in overall_rows:
        r["rank"] = rank_map.get(r["model"], -1)

    if len(model_cfgs) >= 2:
        m1 = model_cfgs[0]["name"]
        m2 = model_cfgs[1]["name"]
        ref1 = next((r for r in overall_rows if r["model"] == m1), None)
        ref2 = next((r for r in overall_rows if r["model"] == m2), None)
        if ref1 is not None and ref2 is not None:
            diff_row = {
                "comparison": f"{m2} - {m1}",
                "impossible_ratio_diff": ref2["impossible_ratio"] - ref1["impossible_ratio"],
                "avg_physics_violation_score_diff": ref2["avg_physics_violation_score"] - ref1["avg_physics_violation_score"],
                "any_violation_ratio_diff": ref2["any_violation_ratio"] - ref1["any_violation_ratio"],
                "cusp_trigger_ratio_diff": ref2["cusp_trigger_ratio"] - ref1["cusp_trigger_ratio"],
                "lat_acc_trigger_ratio_diff": ref2["lat_acc_trigger_ratio"] - ref1["lat_acc_trigger_ratio"],
                "acc_trigger_ratio_diff": ref2["acc_trigger_ratio"] - ref1["acc_trigger_ratio"],
                "jerk_trigger_ratio_diff": ref2["jerk_trigger_ratio"] - ref1["jerk_trigger_ratio"],
                "curvature_rate_trigger_ratio_diff": ref2["curvature_rate_trigger_ratio"] - ref1["curvature_rate_trigger_ratio"],
                "kinematic_inconsistency_trigger_ratio_diff": (
                    ref2["kinematic_inconsistency_trigger_ratio"] - ref1["kinematic_inconsistency_trigger_ratio"]
                ),
                "winner_by_impossible_ratio": m1 if ref1["impossible_ratio"] <= ref2["impossible_ratio"] else m2,
                "winner_by_violation_score": (
                    m1
                    if ref1["avg_physics_violation_score"] <= ref2["avg_physics_violation_score"]
                    else m2
                ),
            }
            comparison_rows = [diff_row]
        else:
            comparison_rows = []
    else:
        comparison_rows = []

    write_csv(os.path.join(args.output_dir, "overall_summary.csv"), overall_rows)
    write_csv(os.path.join(args.output_dir, "per_scenario_summary.csv"), per_scenario_rows)
    write_csv(os.path.join(args.output_dir, "per_violation_summary.csv"), per_violation_rows)
    write_csv(os.path.join(args.output_dir, "scenario_violation_matrix.csv"), scenario_matrix_rows)
    write_csv(os.path.join(args.output_dir, "per_trajectory_metrics.csv"), all_per_traj_rows)
    write_csv(os.path.join(args.output_dir, "topk_cases.csv"), topk_case_rows)
    if comparison_rows:
        write_csv(os.path.join(args.output_dir, "model_comparison.csv"), comparison_rows)

    thresholds_out = {
        "threshold_mode": threshold_obj["mode"],
        "thresholds": thresholds,
        "threshold_meta": threshold_obj.get("meta", {}),
        "args": {
            "threshold_quantile": args.threshold_quantile,
            "threshold_margin": args.threshold_margin,
            "cusp_quantile": args.cusp_quantile,
            "cusp_margin": args.cusp_margin,
            "lat_acc_min_speed_mps": args.lat_acc_min_speed_mps,
            "curvature_rate_min_speed_mps": args.curvature_rate_min_speed_mps,
            "kinematic_min_speed_mps": args.kinematic_min_speed_mps,
            "hard_multiplier": args.hard_multiplier,
            "hard_min_consecutive": args.hard_min_consecutive,
            "force_gt_feasible": bool(args.force_gt_feasible),
        },
    }
    with open(os.path.join(args.output_dir, "thresholds_used.json"), "w") as f:
        json.dump(thresholds_out, f, indent=2)

    summary_json = {
        "config": {
            "data_path": args.data_path,
            "data_type": args.data_type,
            "fps": args.fps,
            "dt": args.dt,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "output_dir": args.output_dir,
            "top_k": args.top_k,
            "include_gt_baseline": args.include_gt_baseline,
            "model1": model_cfgs[0],
            "model2": model_cfgs[1],
            "code_shapes": code_shapes,
        },
        "overall": overall_rows,
        "comparison": comparison_rows,
        "scenarios": scenario_names,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    generate_summary_figures(
        overall_rows=overall_rows,
        per_scenario_rows=per_scenario_rows,
        per_violation_rows=per_violation_rows,
        scenario_names=scenario_names,
        save_dir=args.output_dir,
    )

    print_overall_brief(overall_rows)
    print("=" * 96)
    print(f"Saved outputs to: {args.output_dir}")
    print("Key files:")
    print(f"  - {os.path.join(args.output_dir, 'overall_summary.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'per_scenario_summary.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'per_violation_summary.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'scenario_violation_matrix.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'thresholds_used.json')}")


if __name__ == "__main__":
    main()

# python rvq_transformer_vehdyn/eval_physical_feasibility.py \
#   --data-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --model1-path rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_kin_0311/pred_rvq_taae_model.pth \
#   --model1-type taae --model1-name taae \
#   --model2-path rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_accint_0423/pred_rvq_accint_model.pth \
#   --model2-type accint --model2-name accint \
#   --threshold-mode quantile --threshold-quantile 99.5 --threshold-margin 1.1 \
#   --top-k 3 --include-gt-baseline \
#   --output-dir rvq_transformer_vehdyn/work_dirs/tokenizer/physics_eval
