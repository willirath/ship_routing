#!/usr/bin/env python
"""ONE-OFF top-up of under-represented replicas for cost_001.

8/12/16/20 kn ended up with unequal replica counts because speeds were added in
stages (8/12 ran in all 6 submissions, 16 in 4, 20 in only 2). This script
resubmits the journeys of *one* speed under a fresh submission ID so the
per-speed coverage can be balanced up to the 12 kn level. Baseline and
no-currents forcings are selected via --no-currents.

Disposable: run once via submit_cost_topup.sh, then this script,
run_cost_topup.job and submit_cost_topup.sh can be deleted. See
plans/topup-underrepresented-replicas.md.

Usage
-----
    python topup_cost.py --speed 20 --submission-id 3 --execution nesh-prod-40
    python topup_cost.py --speed 8  --submission-id 4 --no-currents --execution nesh-prod-40
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from experiment_params import (
    ALL_JOURNEYS,
    FORCING_NO_CURRENTS,
    make_production_configs,
)
from ship_routing.htc import run_tuning_sweep as run_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

N_REPLICAS = 5


@click.command()
@click.option(
    "--speed",
    type=float,
    required=True,
    help="Speed in knots to top up (e.g. 8, 16, 20).",
)
@click.option(
    "--submission-id",
    type=int,
    required=True,
    help="Fresh submission ID; must be unique per (speed, forcing). Originals "
    "used 1/2, so use 3, 4, ... and never reuse an ID for the same speed+forcing.",
)
@click.option(
    "--no-currents",
    is_flag=True,
    help="Use the no-currents forcing (ablation) instead of baseline.",
)
@click.option("--execution", default="nesh-prod-40", help="Execution environment.")
@click.option("--dry-run", is_flag=True, help="Build configs but don't run.")
def main(
    speed: float, submission_id: int, no_currents: bool, execution: str, dry_run: bool
) -> None:
    """Resubmit only the journeys of a single speed under a fresh submission ID."""
    journeys = [j for j in ALL_JOURNEYS if j.speed_knots == speed]
    if not journeys:
        raise click.UsageError(
            f"No journeys at {speed:g} kn; valid speeds: "
            f"{sorted({j.speed_knots for j in ALL_JOURNEYS})}"
        )
    forcing_kwargs = {"forcing": FORCING_NO_CURRENTS} if no_currents else {}
    stem = "cost_no_currents" if no_currents else "cost"

    configs = make_production_configs(
        journeys=journeys,
        n_replicas=N_REPLICAS,
        submission_id=submission_id,
        **forcing_kwargs,
    )
    logger.info(
        f"Top-up: {len(journeys)} journeys ({speed:g} kn) x {N_REPLICAS} replicas "
        f"= {len(configs)} configs, {stem}, submission {submission_id}"
    )

    if dry_run:
        for c in configs[:5]:
            logger.info(
                f"  {c.journey.name} {c.journey.time_start} "
                f"{c.journey.speed_knots}kn seed={c.hyper.random_seed}"
            )
        logger.info("Dry run — not executing")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("results") / f"{stem}_sub{submission_id:02d}_{timestamp}.msgpack"
    run_dir = (
        Path("runinfo") / f"{stem}_topup_sub{submission_id:02d}_{timestamp}"
    ).resolve()
    logger.info(f"Output: {output_path}")
    run_sweep(configs, execution, output_path, run_dir=run_dir)


if __name__ == "__main__":
    main()
