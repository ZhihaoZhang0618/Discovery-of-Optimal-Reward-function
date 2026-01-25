import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EVENT_PREFIX = "events.out.tfevents"


@dataclass(frozen=True)
class RunKey:
    case: str
    algo: str
    reward_mode: str
    seed: int


@dataclass
class RunTrend:
    run_dir: str
    key: RunKey
    timestamp: int
    tag: str
    count: int
    step_min: int
    step_max: int
    first_mean: float
    last_mean: float
    mean_all: float
    time_weighted_mean: float
    slope_per_1e5_steps: float
    max_value: float


def iter_event_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.startswith(EVENT_PREFIX):
                yield os.path.join(root, name)


def parse_run_key(rel_run_dir: str) -> Optional[Tuple[RunKey, int]]:
    # <case>__<algo>__<reward_mode>__<seed>__<timestamp>
    parts = rel_run_dir.split("/")
    leaf = parts[-1]
    leaf_parts = leaf.split("__")
    if len(leaf_parts) < 5:
        return None

    reward_mode = leaf_parts[-3]
    if reward_mode not in {"env", "learned"}:
        return None

    try:
        seed = int(leaf_parts[-2])
        timestamp = int(leaf_parts[-1])
    except ValueError:
        return None

    algo = leaf_parts[-4]
    case_leaf = "__".join(leaf_parts[:-4])
    case_prefix = "/".join(parts[:-1])
    case = f"{case_prefix}/{case_leaf}" if case_prefix else case_leaf

    return RunKey(case=case, algo=algo, reward_mode=reward_mode, seed=seed), timestamp


def _mean_or_nan(xs: np.ndarray) -> float:
    return float(np.mean(xs)) if xs.size else float("nan")


def compute_trend(
    event_file: str,
    tag: str,
    first_frac: float,
    last_frac: float,
) -> Optional[Tuple[int, int, int, float, float, float, float, float, float]]:
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None

    events = ea.Scalars(tag)
    if not events:
        return None

    steps = np.asarray([int(e.step) for e in events], dtype=np.int64)
    values = np.asarray([float(e.value) for e in events], dtype=np.float64)

    # Ensure monotonic steps for integration/regression; if not, sort.
    if steps.size >= 2 and np.any(steps[1:] < steps[:-1]):
        order = np.argsort(steps)
        steps = steps[order]
        values = values[order]

    count = int(values.size)
    step_min = int(steps[0])
    step_max = int(steps[-1])

    first_n = max(1, int(math.ceil(count * first_frac)))
    last_n = max(1, int(math.ceil(count * last_frac)))

    first_mean = _mean_or_nan(values[:first_n])
    last_mean = _mean_or_nan(values[-last_n:])
    mean_all = _mean_or_nan(values)
    max_value = float(np.max(values))

    # Time-weighted mean via trapezoidal integral over steps.
    if count >= 2 and step_max > step_min:
        auc = float(np.trapz(values, steps))
        time_weighted_mean = auc / float(step_max - step_min)
    else:
        time_weighted_mean = mean_all

    # Slope via simple linear regression value ~ step.
    if count >= 2 and np.var(steps) > 0:
        slope = float(np.polyfit(steps.astype(np.float64), values, deg=1)[0])
        slope_per_1e5 = slope * 1e5
    else:
        slope_per_1e5 = 0.0

    return (
        count,
        step_min,
        step_max,
        first_mean,
        last_mean,
        mean_all,
        time_weighted_mean,
        slope_per_1e5,
        max_value,
    )


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="runs/ directory (or a specific run dir)")
    p.add_argument("--tag", default="charts/episodic_return")
    p.add_argument("--first-frac", type=float, default=0.2, help="Fraction of points for early-phase mean")
    p.add_argument("--last-frac", type=float, default=0.2, help="Fraction of points for late-phase mean")
    p.add_argument("--case-regex", default=None)
    p.add_argument("--algo-regex", default=None)
    args = p.parse_args()

    if not (0 < args.first_frac <= 1.0) or not (0 < args.last_frac <= 1.0):
        raise SystemExit("--first-frac/--last-frac must be in (0, 1]")

    case_re = re.compile(args.case_regex) if args.case_regex else None
    algo_re = re.compile(args.algo_regex) if args.algo_regex else None

    best_by_key: Dict[RunKey, RunTrend] = {}

    for event_file in iter_event_files(args.path):
        run_dir = os.path.dirname(event_file)
        rel_run_dir = os.path.relpath(run_dir, args.path) if os.path.isdir(args.path) else os.path.relpath(run_dir)

        parsed = parse_run_key(rel_run_dir)
        if parsed is None:
            continue
        key, ts = parsed
        if case_re and not case_re.search(key.case):
            continue
        if algo_re and not algo_re.search(key.algo):
            continue

        trend = compute_trend(event_file, args.tag, args.first_frac, args.last_frac)
        if trend is None:
            continue
        (
            count,
            step_min,
            step_max,
            first_mean,
            last_mean,
            mean_all,
            time_weighted_mean,
            slope_per_1e5_steps,
            max_value,
        ) = trend

        summary = RunTrend(
            run_dir=os.path.join(args.path, rel_run_dir) if os.path.isdir(args.path) else run_dir,
            key=key,
            timestamp=ts,
            tag=args.tag,
            count=count,
            step_min=step_min,
            step_max=step_max,
            first_mean=first_mean,
            last_mean=last_mean,
            mean_all=mean_all,
            time_weighted_mean=time_weighted_mean,
            slope_per_1e5_steps=slope_per_1e5_steps,
            max_value=max_value,
        )

        prev = best_by_key.get(key)
        if prev is None or summary.timestamp > prev.timestamp:
            best_by_key[key] = summary

    if not best_by_key:
        print("No matching runs found.")
        return 2

    # Group across seeds by (case, algo, reward_mode)
    grouped: Dict[Tuple[str, str, str], List[RunTrend]] = {}
    for s in best_by_key.values():
        grouped.setdefault((s.key.case, s.key.algo, s.key.reward_mode), []).append(s)

    # Header
    print(
        "case\talgo\treward_mode\tseeds\tcount\tstep_min\tstep_max\t"
        "first_mean\tlast_mean\tmean_all\ttime_weighted_mean\tslope_per_1e5_steps\tmax"
    )

    for group_key in sorted(grouped.keys()):
        case, algo, reward_mode = group_key
        runs = sorted(grouped[group_key], key=lambda r: r.key.seed)
        seeds = [r.key.seed for r in runs]

        # For cross-seed aggregation, we aggregate means across seeds.
        firsts = [r.first_mean for r in runs]
        lasts = [r.last_mean for r in runs]
        means_all = [r.mean_all for r in runs]
        tw_means = [r.time_weighted_mean for r in runs]
        slopes = [r.slope_per_1e5_steps for r in runs]
        max_overall = max(r.max_value for r in runs)

        # Use the average of step_min/max and counts for reporting.
        count_avg = int(round(sum(r.count for r in runs) / len(runs)))
        step_min_avg = int(round(sum(r.step_min for r in runs) / len(runs)))
        step_max_avg = int(round(sum(r.step_max for r in runs) / len(runs)))

        print(
            f"{case}\t{algo}\t{reward_mode}\t{seeds}\t{count_avg}\t{step_min_avg}\t{step_max_avg}\t"
            f"{np.mean(firsts):.6g}\t{np.mean(lasts):.6g}\t{np.mean(means_all):.6g}\t{np.mean(tw_means):.6g}\t{np.mean(slopes):.6g}\t{max_overall:.6g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
