#!/usr/bin/env bash
for _ in {1..10}; do  # 3 realisations
	for mm in $(seq -w 1 12); do  # each month
		for spd in 8.0 10.0 12.0; do  # three speeds
		       	sbatch --export="TIME_START=\"2021-${mm}-01T00:00:00\",SPEED_KNOTS=\"${spd}\",JOURNEY_NAME=\"Atlantic\"" run_experiment.job;
		done
	done
done
