# Splendor/tb_report.py
"""
Standalone TensorBoard scalar reporter. Parses tfevents files and prints a
compact per-tag summary (first/last/min/max + recent-window mean + trend), so
runs can be analyzed from the values directly rather than by eyeballing graphs.

Usage:
    python tb_report.py                  # newest run under the default logdir
    python tb_report.py <run_dir>        # a specific run directory
    python tb_report.py --logdir <dir>   # newest run under <dir>
    python tb_report.py --all            # every run under the default logdir

Default logdir: RL/saved_files/tensorboard_logs (relative to this file).
Reads files on disk, so it works on the host even though training runs in the
devcontainer (shared workspace).
"""

import os
import sys
import glob

import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGDIR = os.path.join(HERE, "RL", "saved_files", "tensorboard_logs")


def _scalars_for_run(run_dir):
    """tag -> list of (step, value), sorted by step."""
    series = {}
    for ev_file in glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"),
                             recursive=True):
        for event in summary_iterator(ev_file):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    series.setdefault(value.tag, []).append(
                        (event.step, value.simple_value))
                elif value.HasField('tensor'):
                    # TF2 scalars are stored as DT_FLOAT tensor events; decode
                    # robustly (covers float_val and tensor_content encodings).
                    try:
                        arr = tf.make_ndarray(value.tensor)
                        if arr.ndim == 0:
                            series.setdefault(value.tag, []).append(
                                (event.step, float(arr)))
                    except Exception:
                        pass
    for tag in series:
        series[tag].sort(key=lambda t: t[0])
    return series


def _report_run(run_dir, window=10):
    series = _scalars_for_run(run_dir)
    print(f"\n=== {os.path.basename(run_dir.rstrip(os.sep))} ===")
    if not series:
        print("  (no scalar data)")
        return
    name_w = max(len(t) for t in series)
    header = (f"  {'tag':<{name_w}}  {'n':>4}  {'first':>10}  {'last':>10}  "
              f"{'min':>10}  {'max':>10}  {'last%d_mean' % window:>12}  trend")
    print(header)
    for tag in sorted(series):
        pts = series[tag]
        steps, vals = zip(*pts)
        first, last = vals[0], vals[-1]
        recent = vals[-window:]
        recent_mean = sum(recent) / len(recent)
        trend = "flat"
        if last > first * 1.05 or (first == 0 and last > 0.01):
            trend = "up"
        elif last < first * 0.95 or (first == 0 and last < -0.01):
            trend = "down"
        print(f"  {tag:<{name_w}}  {len(pts):>4}  {first:>10.4f}  {last:>10.4f}  "
              f"{min(vals):>10.4f}  {max(vals):>10.4f}  {recent_mean:>12.4f}  {trend}")


def _newest_run(logdir):
    runs = [d for d in glob.glob(os.path.join(logdir, "*")) if os.path.isdir(d)]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def main(argv):
    logdir = DEFAULT_LOGDIR
    show_all = False
    explicit_run = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--logdir":
            logdir = argv[i + 1]; i += 2
        elif a == "--all":
            show_all = True; i += 1
        else:
            explicit_run = a; i += 1

    if explicit_run:
        _report_run(explicit_run)
        return

    if not os.path.isdir(logdir):
        print(f"Logdir not found: {logdir}")
        return

    if show_all:
        runs = sorted((d for d in glob.glob(os.path.join(logdir, "*"))
                       if os.path.isdir(d)), key=os.path.getmtime)
        if not runs:
            print(f"No runs under {logdir}")
        for r in runs:
            _report_run(r)
    else:
        newest = _newest_run(logdir)
        if newest is None:
            print(f"No runs under {logdir}")
        else:
            _report_run(newest)


if __name__ == "__main__":
    main(sys.argv[1:])
