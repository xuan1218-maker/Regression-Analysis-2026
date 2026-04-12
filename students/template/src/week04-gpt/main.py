"""Main entrypoint for week04-gpt teaching template.

main.py stays intentionally thin:
- load config
- call benchmark helpers
- print results
"""

from __future__ import annotations

from config import get_default_config
from utils.benchmark import run_scenario
from utils.display import print_rows, print_scenario_header


def main() -> None:
    experiment = get_default_config()

    for scenario in (experiment.low_dim, experiment.high_dim):
        print_scenario_header(
            name=scenario.name,
            n_samples=scenario.n_samples,
            n_features=scenario.n_features,
        )
        rows = run_scenario(cfg=scenario, solver_cfg=experiment.solver)
        print_rows(rows)


if __name__ == "__main__":
    main()
