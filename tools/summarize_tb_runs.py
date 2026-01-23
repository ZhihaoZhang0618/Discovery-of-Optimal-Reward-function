import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class RunKey:
    case: str
    algo: str
    reward_mode: str
    seed: int


@dataclass
class RunSummary:
    run_dir: str
    key: RunKey
    timestamp: int
    count: int
    last_step: int
    last_value: float
    mean_last_n: float
    max_value: float


EVENT_PREFIX = "events.out.tfevents"


def iter_event_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.startswith(EVENT_PREFIX):
                yield os.path.join(root, name)


def parse_run_key(rel_run_dir: str) -> Optional[Tuple[RunKey, int]]:
    """Parse run directory name into (RunKey, timestamp).

    Expected folder name pattern:
      <case>__<algo>__<reward_mode>__<seed>__<timestamp>

    For nested runs (e.g. runs/PyFlyt/QuadX-Waypoints-v4__...), `rel_run_dir`
    can contain slashes; we treat the parent folder(s) as part of `case`.
    """
    # rel_run_dir like:
    #   CartPole-v1__dqn__env__1__1768633906
    #   PyFlyt/QuadX-Waypoints-v4__sac_continuous_action__env__1__1768648320
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


def load_scalar_summary(event_file: str, tag: str, last_n: int) -> Optional[Tuple[int, int, float, float, float]]:
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    if not events:
        return None

    steps = [e.step for e in events]
    values = [float(e.value) for e in events]

    count = len(values)
    last_step = steps[-1]
    last_value = values[-1]
    max_value = max(values)
    n = min(last_n, count)
    mean_last_n = sum(values[-n:]) / n
    return count, last_step, last_value, mean_last_n, max_value


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
    p.add_argument("--last-n", type=int, default=20)
    p.add_argument("--case-regex", default=None, help="Regex to filter RunKey.case")
    p.add_argument("--algo-regex", default=None, help="Regex to filter RunKey.algo")
    args = p.parse_args()

    case_re = re.compile(args.case_regex) if args.case_regex else None
    algo_re = re.compile(args.algo_regex) if args.algo_regex else None

    best_by_key: Dict[RunKey, RunSummary] = {}

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

        scalar = load_scalar_summary(event_file, args.tag, args.last_n)
        if scalar is None:
            continue
        count, last_step, last_value, mean_last_n, max_value = scalar

        summary = RunSummary(
            run_dir=os.path.join(args.path, rel_run_dir) if os.path.isdir(args.path) else run_dir,
            key=key,
            timestamp=ts,
            count=count,
            last_step=last_step,
            last_value=last_value,
            mean_last_n=mean_last_n,
            max_value=max_value,
        )

        prev = best_by_key.get(key)
        if prev is None or summary.timestamp > prev.timestamp:
            best_by_key[key] = summary

    if not best_by_key:
        print("No matching runs found.")
        return 2

    # Group across seeds by (case, algo, reward_mode)
    grouped: Dict[Tuple[str, str, str], List[RunSummary]] = {}
    for s in best_by_key.values():
        grouped.setdefault((s.key.case, s.key.algo, s.key.reward_mode), []).append(s)

    for group_key in sorted(grouped.keys()):
        case, algo, reward_mode = group_key
        runs = sorted(grouped[group_key], key=lambda r: r.key.seed)
        values = [r.mean_last_n for r in runs]
        mu, sd = mean_std(values)
        seeds = [r.key.seed for r in runs]
        max_overall = max(r.max_value for r in runs)
        print(
            f"{case}\t{algo}\t{reward_mode}\tseeds={seeds}\tmean_last{args.last_n}={mu:.3f}\tstd={sd:.3f}\tmax={max_overall:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
