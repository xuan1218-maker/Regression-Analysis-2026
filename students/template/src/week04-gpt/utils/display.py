"""Display helpers for concise terminal output."""

from __future__ import annotations

from typing import Iterable


def print_scenario_header(name: str, n_samples: int, n_features: int) -> None:
    print(f"\n=== Scenario: {name} (N={n_samples}, P={n_features}) ===")


def print_rows(rows: Iterable[tuple[str, float, float]]) -> None:
    print(f"{'solver':28s} | {'time(s)':>10s} | {'mse':>12s}")
    print("-" * 58)
    for solver_name, sec, err in rows:
        print(f"{solver_name:28s} | {sec:10.4f} | {err:12.6f}")
