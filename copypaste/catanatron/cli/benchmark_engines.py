from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from rich.table import Table

from catanatron.cli.cli_players import parse_cli_string
from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points
from catanatron.utils import ensure_dir

SEAT_COLORS: Tuple[Color, ...] = (
    Color.RED,
    Color.BLUE,
    Color.ORANGE,
    Color.WHITE,
)


@dataclass(frozen=True)
class BenchmarkTask:
    candidate_spec: str
    baseline_spec: str
    seat_index: int
    seed: int
    catan_map: str
    vps_to_win: int
    discard_limit: int


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _seat_label(seat_index: int) -> str:
    color = SEAT_COLORS[seat_index]
    return f"P{seat_index + 1}_{color.value}"


def _lineup_string(candidate_spec: str, baseline_spec: str, seat_index: int) -> str:
    lineup = [baseline_spec] * 4
    lineup[seat_index] = candidate_spec
    return ",".join(lineup)


def _binomial_ci95(wins: int, games: int) -> Tuple[float, float, float]:
    if games <= 0:
        return 0.0, 0.0, 0.0
    p = wins / games
    margin = 1.96 * math.sqrt(max(0.0, p * (1.0 - p)) / games)
    return p, max(0.0, p - margin), min(1.0, p + margin)


def _run_single_task(task: BenchmarkTask) -> Dict[str, Any]:
    random.seed(task.seed)

    lineup = _lineup_string(task.candidate_spec, task.baseline_spec, task.seat_index)
    players = parse_cli_string(lineup)
    game = Game(
        players,
        discard_limit=task.discard_limit,
        vps_to_win=task.vps_to_win,
        catan_map=build_map(task.catan_map),
    )
    game.play()

    winner = game.winning_color()
    candidate_color = SEAT_COLORS[task.seat_index]
    baseline_colors = tuple(c for i, c in enumerate(SEAT_COLORS) if i != task.seat_index)

    candidate_vp = int(get_actual_victory_points(game.state, candidate_color))
    vps_by_color = {
        color.value: int(get_actual_victory_points(game.state, color))
        for color in SEAT_COLORS
    }

    return {
        "candidate_spec": task.candidate_spec,
        "baseline_spec": task.baseline_spec,
        "seat_index": task.seat_index,
        "seed": task.seed,
        "winner_color": winner.value if winner is not None else None,
        "candidate_color": candidate_color.value,
        "baseline_colors": [c.value for c in baseline_colors],
        "candidate_vp": candidate_vp,
        "turns": int(game.state.num_turns),
        "ticks": int(len(game.state.action_records)),
        "vps_by_color": vps_by_color,
    }


def _build_tasks(
    candidates: Sequence[str],
    baseline: str,
    games_per_seat: int,
    seed_start: int,
    catan_map: str,
    vps_to_win: int,
    discard_limit: int,
) -> List[BenchmarkTask]:
    # Common-random-numbers schedule:
    # each candidate receives identical seeds per seat for better comparability.
    seeds_by_seat: List[List[int]] = []
    cursor = seed_start
    for _seat in range(4):
        seat_seeds = [cursor + i for i in range(games_per_seat)]
        cursor += games_per_seat
        seeds_by_seat.append(seat_seeds)

    tasks: List[BenchmarkTask] = []
    for candidate in candidates:
        for seat_index in range(4):
            for seed in seeds_by_seat[seat_index]:
                tasks.append(
                    BenchmarkTask(
                        candidate_spec=candidate,
                        baseline_spec=baseline,
                        seat_index=seat_index,
                        seed=seed,
                        catan_map=catan_map,
                        vps_to_win=vps_to_win,
                        discard_limit=discard_limit,
                    )
                )
    return tasks


def _results_iterator(
    tasks: Sequence[BenchmarkTask],
    parallel: bool,
    workers: int | None,
) -> Iterable[Dict[str, Any]]:
    if not parallel:
        for task in tasks:
            yield _run_single_task(task)
        return

    process_count = workers if workers is not None else cpu_count()
    process_count = max(1, int(process_count))
    chunk_size = max(1, len(tasks) // (process_count * 20))
    with Pool(processes=process_count) as pool:
        for result in pool.imap_unordered(_run_single_task, tasks, chunksize=chunk_size):
            yield result


def _format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _build_markdown_report(
    label: str,
    baseline: str,
    catan_map: str,
    vps_to_win: int,
    discard_limit: int,
    games_per_seat: int,
    seed_start: int,
    parallel: bool,
    workers: int | None,
    elapsed_seconds: float,
    overall_rows: Sequence[Dict[str, Any]],
    seat_rows: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Benchmark Report — {label}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Baseline: `{baseline}`")
    lines.append(f"- Map: `{catan_map}`")
    lines.append(f"- VPS to win: `{vps_to_win}`")
    lines.append(f"- Discard limit: `{discard_limit}`")
    lines.append(f"- Games per seat: `{games_per_seat}`")
    lines.append(f"- Total seats: `4`")
    lines.append(f"- Seed start: `{seed_start}`")
    lines.append(f"- Parallel: `{parallel}`")
    lines.append(f"- Workers: `{workers}`")
    lines.append(f"- Runtime: `{elapsed_seconds:.2f}s`")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| Candidate | Games | Candidate Wins | Candidate Win Rate | 95% CI | Baseline Per-Player Win Rate | Advantage | Avg Candidate VP | Avg Turns | Avg Ticks |"
    )
    lines.append(
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"
    )
    for row in overall_rows:
        lines.append(
            f"| `{row['candidate']}` | {row['games']} | {row['candidate_wins']} | "
            f"{row['candidate_win_rate_pct']} | "
            f"[{row['ci_low_pct']}, {row['ci_high_pct']}] | "
            f"{row['baseline_per_player_win_rate_pct']} | "
            f"{row['advantage_vs_baseline_player_pct']} | "
            f"{row['avg_candidate_vp']:.3f} | "
            f"{row['avg_turns']:.2f} | {row['avg_ticks']:.2f} |"
        )
    lines.append("")

    lines.append("## By Seat")
    lines.append("")
    lines.append(
        "| Candidate | Seat | Games | Candidate Wins | Candidate Win Rate | Avg Candidate VP | Avg Turns | Avg Ticks |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in seat_rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['seat']}` | {row['games']} | {row['candidate_wins']} | "
            f"{row['candidate_win_rate_pct']} | {row['avg_candidate_vp']:.3f} | "
            f"{row['avg_turns']:.2f} | {row['avg_ticks']:.2f} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append(
        "- Candidate win rate is measured as the single candidate player's win rate in a 4-player game with 3 baseline copies."
    )
    lines.append(
        "- Baseline per-player win rate is baseline total wins divided by 3 baseline seats."
    )
    lines.append(
        "- Advantage is candidate win rate minus baseline per-player win rate."
    )
    lines.append("")
    lines.append(
        "- For stronger claims, increase games per seat (e.g., 500 or 1000)."
    )
    lines.append(
        "- This benchmark uses common-random-number seeds per seat to improve comparability across candidates."
    )
    lines.append("")
    return "\n".join(lines)


@click.command()
@click.option(
    "--candidates",
    required=True,
    help="Comma-separated candidate engine specs. Example: GT,SAB:2:true",
)
@click.option(
    "--baseline",
    required=True,
    help="Baseline engine spec used for the 3 opponent seats. Example: AB:2:true",
)
@click.option(
    "--games-per-seat",
    default=250,
    show_default=True,
    type=click.IntRange(1, None),
    help="Games to run for each seat position (P1..P4) per candidate.",
)
@click.option(
    "--seed-start",
    default=1000,
    show_default=True,
    type=int,
    help="Starting seed for common-random-number schedules.",
)
@click.option(
    "--config-map",
    "catan_map",
    default="TOURNAMENT",
    show_default=True,
    type=click.Choice(["BASE", "MINI", "TOURNAMENT"], case_sensitive=False),
    help="Map type for all benchmark games.",
)
@click.option(
    "--config-vps-to-win",
    "vps_to_win",
    default=10,
    show_default=True,
    type=click.IntRange(3, None),
    help="Victory points required to win.",
)
@click.option(
    "--config-discard-limit",
    "discard_limit",
    default=7,
    show_default=True,
    type=click.IntRange(2, None),
    help="Discard limit used by the game.",
)
@click.option(
    "--parallel/--no-parallel",
    default=False,
    show_default=True,
    help="Run benchmark games in parallel with multiprocessing.",
)
@click.option(
    "--workers",
    default=None,
    type=click.IntRange(1, None),
    help="Worker processes for --parallel (default: CPU count).",
)
@click.option(
    "--label",
    default=None,
    help="Optional run label for output files. Default: auto timestamp label.",
)
@click.option(
    "--out-dir",
    default="benchmark_results",
    show_default=True,
    help="Directory for benchmark output files.",
)
@click.option(
    "--md-archive-dir",
    default="benchmark_md_by_date",
    show_default=True,
    help="Separate folder to archive benchmark markdown reports by benchmark end date.",
)
def benchmark_engines(
    candidates: str,
    baseline: str,
    games_per_seat: int,
    seed_start: int,
    catan_map: str,
    vps_to_win: int,
    discard_limit: int,
    parallel: bool,
    workers: int | None,
    label: str | None,
    out_dir: str,
    md_archive_dir: str,
):
    """
    Benchmark candidate engines against a baseline field.

    For each candidate:
      - Run 4 seat variants:
        P1=candidate vs baseline x3,
        P2=candidate vs baseline x3,
        P3=candidate vs baseline x3,
        P4=candidate vs baseline x3.
      - Each seat uses GAMES_PER_SEAT games.
      - Common-random-number seeds are reused per seat across candidates.
    """

    console = Console()
    candidate_specs = _split_csv(candidates)
    if not candidate_specs:
        raise click.BadParameter("No candidates provided.", param_hint="--candidates")

    if games_per_seat < 1:
        raise click.BadParameter(
            "--games-per-seat must be >= 1", param_hint="--games-per-seat"
        )

    run_label = label or f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(out_dir)
    out_path = Path(out_dir)

    tasks = _build_tasks(
        candidates=candidate_specs,
        baseline=baseline,
        games_per_seat=games_per_seat,
        seed_start=seed_start,
        catan_map=catan_map,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
    )
    total_games = len(tasks)

    console.print(
        f"[green]Running benchmark[/green] label={run_label} "
        f"(candidates={len(candidate_specs)}, games={total_games}, "
        f"parallel={parallel}, workers={workers or cpu_count() if parallel else 1})"
    )

    stats: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(
        lambda: {
            "games": 0.0,
            "candidate_wins": 0.0,
            "baseline_total_wins": 0.0,
            "candidate_vp_sum": 0.0,
            "turns_sum": 0.0,
            "ticks_sum": 0.0,
        }
    )

    started = time.time()
    iterator = _results_iterator(tasks, parallel=parallel, workers=workers)
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Benchmarking...", total=total_games)
        for result in iterator:
            candidate = str(result["candidate_spec"])
            seat = int(result["seat_index"])
            winner = result["winner_color"]
            candidate_color = str(result["candidate_color"])
            baseline_colors = set(result["baseline_colors"])

            key = (candidate, seat)
            row = stats[key]
            row["games"] += 1.0
            if winner == candidate_color:
                row["candidate_wins"] += 1.0
            elif winner in baseline_colors:
                row["baseline_total_wins"] += 1.0

            row["candidate_vp_sum"] += float(result["candidate_vp"])
            row["turns_sum"] += float(result["turns"])
            row["ticks_sum"] += float(result["ticks"])
            progress.update(bar, advance=1)

    elapsed = time.time() - started

    # Build per-seat rows
    seat_rows: List[Dict[str, Any]] = []
    for candidate in candidate_specs:
        for seat in range(4):
            row = stats[(candidate, seat)]
            games = int(row["games"])
            cand_wins = int(row["candidate_wins"])
            cand_wr, _, _ = _binomial_ci95(cand_wins, games)
            seat_rows.append(
                {
                    "candidate": candidate,
                    "seat": _seat_label(seat),
                    "games": games,
                    "candidate_wins": cand_wins,
                    "candidate_win_rate": cand_wr,
                    "candidate_win_rate_pct": _format_pct(cand_wr),
                    "avg_candidate_vp": row["candidate_vp_sum"] / games if games else 0.0,
                    "avg_turns": row["turns_sum"] / games if games else 0.0,
                    "avg_ticks": row["ticks_sum"] / games if games else 0.0,
                }
            )

    # Build overall rows
    overall_rows: List[Dict[str, Any]] = []
    for candidate in candidate_specs:
        per_seat = [stats[(candidate, seat)] for seat in range(4)]
        games = int(sum(r["games"] for r in per_seat))
        cand_wins = int(sum(r["candidate_wins"] for r in per_seat))
        baseline_total_wins = int(sum(r["baseline_total_wins"] for r in per_seat))
        p, ci_low, ci_high = _binomial_ci95(cand_wins, games)
        baseline_per_player = (
            (baseline_total_wins / games) / 3.0 if games > 0 else 0.0
        )
        advantage = p - baseline_per_player

        vp_sum = sum(r["candidate_vp_sum"] for r in per_seat)
        turns_sum = sum(r["turns_sum"] for r in per_seat)
        ticks_sum = sum(r["ticks_sum"] for r in per_seat)

        overall_rows.append(
            {
                "candidate": candidate,
                "games": games,
                "candidate_wins": cand_wins,
                "candidate_win_rate": p,
                "candidate_win_rate_pct": _format_pct(p),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_pct": _format_pct(ci_low),
                "ci_high_pct": _format_pct(ci_high),
                "baseline_total_wins": baseline_total_wins,
                "baseline_per_player_win_rate": baseline_per_player,
                "baseline_per_player_win_rate_pct": _format_pct(baseline_per_player),
                "advantage_vs_baseline_player": advantage,
                "advantage_vs_baseline_player_pct": _format_pct(advantage),
                "avg_candidate_vp": vp_sum / games if games else 0.0,
                "avg_turns": turns_sum / games if games else 0.0,
                "avg_ticks": ticks_sum / games if games else 0.0,
            }
        )

    overall_rows.sort(
        key=lambda r: (
            r["advantage_vs_baseline_player"],
            r["candidate_win_rate"],
            r["avg_candidate_vp"],
        ),
        reverse=True,
    )

    seat_rows.sort(key=lambda r: (r["candidate"], r["seat"]))

    # Console summary
    summary_table = Table(title=f"Benchmark Summary - {run_label}")
    summary_table.add_column("Candidate")
    summary_table.add_column("Games", justify="right")
    summary_table.add_column("Wins", justify="right")
    summary_table.add_column("Win Rate", justify="right")
    summary_table.add_column("95% CI", justify="right")
    summary_table.add_column("Base/Player", justify="right")
    summary_table.add_column("Advantage", justify="right")
    summary_table.add_column("Avg VP", justify="right")
    for row in overall_rows:
        summary_table.add_row(
            row["candidate"],
            str(row["games"]),
            str(row["candidate_wins"]),
            row["candidate_win_rate_pct"],
            f"{row['ci_low_pct']}..{row['ci_high_pct']}",
            row["baseline_per_player_win_rate_pct"],
            row["advantage_vs_baseline_player_pct"],
            f"{row['avg_candidate_vp']:.3f}",
        )
    console.print(summary_table)

    # Write files
    summary_json_path = out_path / f"{run_label}_summary.json"
    overall_csv_path = out_path / f"{run_label}_overall.csv"
    seat_csv_path = out_path / f"{run_label}_by_seat.csv"
    report_md_path = out_path / f"{run_label}_report.md"

    payload = {
        "label": run_label,
        "metadata": {
            "baseline": baseline,
            "catan_map": catan_map,
            "vps_to_win": vps_to_win,
            "discard_limit": discard_limit,
            "games_per_seat": games_per_seat,
            "total_games": total_games,
            "seed_start": seed_start,
            "parallel": parallel,
            "workers": workers if parallel else 1,
            "md_archive_dir": md_archive_dir,
            "elapsed_seconds": elapsed,
        },
        "overall": overall_rows,
        "by_seat": seat_rows,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(overall_csv_path, overall_rows)
    _write_csv(seat_csv_path, seat_rows)
    report_md = _build_markdown_report(
        label=run_label,
        baseline=baseline,
        catan_map=catan_map,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        games_per_seat=games_per_seat,
        seed_start=seed_start,
        parallel=parallel,
        workers=(workers if parallel else 1),
        elapsed_seconds=elapsed,
        overall_rows=overall_rows,
        seat_rows=seat_rows,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    # Also archive only the benchmark markdown report in a separate date-sorted folder.
    finished_epoch = time.time()
    finished_date = time.strftime("%Y-%m-%d", time.localtime(finished_epoch))
    finished_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(finished_epoch))
    md_archive_root = Path(md_archive_dir)
    ensure_dir(str(md_archive_root))
    archive_date_dir = md_archive_root / finished_date
    ensure_dir(str(archive_date_dir))
    archive_md_path = _unique_path(
        archive_date_dir / f"{finished_stamp}__{run_label}_report.md"
    )
    archive_md_path.write_text(report_md, encoding="utf-8")

    console.print(f"[green]Done.[/green] Runtime: {elapsed:.2f}s")
    console.print(f"Summary JSON: {summary_json_path}")
    console.print(f"Overall CSV:  {overall_csv_path}")
    console.print(f"Seat CSV:     {seat_csv_path}")
    console.print(f"Report MD:    {report_md_path}")
    console.print(f"Report MD Archive: {archive_md_path}")


if __name__ == "__main__":
    benchmark_engines()
