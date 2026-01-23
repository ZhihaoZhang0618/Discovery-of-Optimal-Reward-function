import argparse
import os
from dataclasses import dataclass
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class ScalarSummary:
    tag: str
    count: int
    last_step: int
    last_value: float
    mean_last_n: float
    max_value: float


def iter_event_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return

    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.startswith("events.out.tfevents"):
                yield os.path.join(root, name)


def load_scalars(event_file: str, tag: str) -> ScalarSummary | None:
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None

    events = ea.Scalars(tag)
    if not events:
        return None

    steps = [e.step for e in events]
    values = [float(e.value) for e in events]

    last_step = steps[-1]
    last_value = values[-1]
    max_value = max(values)

    return ScalarSummary(
        tag=tag,
        count=len(values),
        last_step=last_step,
        last_value=last_value,
        mean_last_n=sum(values[-20:]) / min(20, len(values)),
        max_value=max_value,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Run dir (e.g., runs/...) or a single event file")
    p.add_argument("--tag", default="charts/episodic_return")
    args = p.parse_args()

    found_any = False
    for event_file in sorted(iter_event_files(args.path)):
        summary = load_scalars(event_file, args.tag)
        if summary is None:
            continue
        found_any = True
        run_dir = os.path.relpath(os.path.dirname(event_file))
        print(f"{run_dir}\t{summary.tag}\tcount={summary.count}\tlast_step={summary.last_step}\tlast={summary.last_value:.3f}\tmean_last20={summary.mean_last_n:.3f}\tmax={summary.max_value:.3f}")

    if not found_any:
        print(f"No scalars with tag={args.tag!r} found under: {args.path}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
