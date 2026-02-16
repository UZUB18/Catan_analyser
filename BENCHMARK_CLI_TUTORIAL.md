# Catan Engine Benchmark CLI Tutorial

This tutorial shows how to run a **full, reproducible benchmark** with the new CLI command:

- `catanatron-benchmark`

It benchmarks one or more candidate engines against a 3-seat baseline field.

If the command is not found in your shell yet, run the module form instead:

```bash
python -m catanatron.cli.benchmark_engines --help
```

Or reinstall editable package to register the script:

```bash
pip install -e .
```

---

## 1) What this benchmark does

For each candidate engine:

- Runs 4 seat variants:
  - P1 = candidate, others = baseline
  - P2 = candidate, others = baseline
  - P3 = candidate, others = baseline
  - P4 = candidate, others = baseline
- Runs `--games-per-seat` games for each seat.
- Uses deterministic seed schedules (starting from `--seed-start`) so runs are reproducible.

> Note: The game engine internally randomizes seating as part of game state init, but this benchmark still rotates candidate color slots and uses fixed seed schedules for comparability.

---

## 2) Quick smoke test (small run)

Run this first to verify everything works:

```bash
catanatron-benchmark ^
  --candidates GT,SAB:2:true ^
  --baseline AB:2:true ^
  --games-per-seat 10 ^
  --seed-start 1000 ^
  --config-map TOURNAMENT ^
  --parallel --workers 6 ^
  --label smoke_gt_sab_vs_ab2
```

What to check:

- Command finishes without errors.
- `benchmark_results/` contains output files.

---

## 3) Full benchmark (recommended)

For a stronger conclusion:

- `--games-per-seat 500` (total = 500 × 4 seats = 2000 games per candidate)

Example:

```bash
catanatron-benchmark ^
  --candidates GT,SAB:2:true,AB:2:true ^
  --baseline AB:2:true ^
  --games-per-seat 500 ^
  --seed-start 50000 ^
  --config-map TOURNAMENT ^
  --config-vps-to-win 10 ^
  --config-discard-limit 7 ^
  --parallel --workers 6 ^
  --label full_500ps_vs_ab2
```

If your PC stays responsive, try `--workers 8`.

---

## 4) Output files

Each run writes:

- `benchmark_results/<label>_summary.json`
- `benchmark_results/<label>_overall.csv`
- `benchmark_results/<label>_by_seat.csv`
- `benchmark_results/<label>_report.md`

Use `<label>_report.md` for the human-readable summary.

Additionally, the benchmark auto-archives only the report MD into:

- `benchmark_md_by_date/<YYYY-MM-DD>/<timestamp>__<label>_report.md`

You can change that folder with `--md-archive-dir`.

---

## 5) How to interpret results

In the summary:

- **Candidate Win Rate** = win rate of the single candidate seat.
- **Baseline Per-Player Win Rate** = baseline total wins divided by 3 baseline seats.
- **Advantage** = Candidate Win Rate − Baseline Per-Player Win Rate.
- **95% CI** = confidence interval for candidate win rate.

Practical rule:

- Prefer candidates with **positive advantage** and stable results across repeated full runs.

---

## 6) Reproducibility checklist

To reproduce exactly:

1. Keep the same command arguments.
2. Keep the same `--seed-start`.
3. Keep the same code version/commit.
4. Keep the same map and game config options.

---

## 7) Suggested benchmark ladder

- **Smoke:** 10 games/seat
- **Draft:** 100 games/seat
- **Decision:** 500+ games/seat
- **Strong claim:** 1000 games/seat

---

## 8) Common examples

### GT only vs AB baseline

```bash
catanatron-benchmark --candidates GT --baseline AB:2:true --games-per-seat 500 --parallel --workers 6 --label gt_vs_ab2_500ps
```

### Compare several candidates vs same baseline

```bash
catanatron-benchmark --candidates GT,SAB:2:true,STAT,WILD --baseline AB:2:true --games-per-seat 300 --parallel --workers 6 --label multi_vs_ab2_300ps
```
