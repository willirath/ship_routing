#!/usr/bin/env bash
# Submit chained SLURM jobs for concurrentablation experiments.

set -e  # Exit on error

# Configuration
BATCHES_PER_SCENARIO=13  # amounts to 13*4000=52000 samples per scenario

# Submit jobs for each scenario
for job_file in run_{baseline,no_currents,no_winds,no_waves}.job; do
    prev_job_id=""

    for batch in $(seq 1 $BATCHES_PER_SCENARIO); do
        # Build sbatch command
        if [ -z "$prev_job_id" ]; then
            # First batch - no dependency
            job_id=$(sbatch --parsable "$job_file")
            echo "Batch $batch of $job_file: Submitted job $job_id (no dependency)"
        else
            # Subsequent batches - depend on previous job
            job_id=$(sbatch --parsable --dependency=afterany:"$prev_job_id" "$job_file")
            echo "Batch $batch of $job_file: Submitted job $job_id (depends on $prev_job_id)"
        fi

        prev_job_id=$job_id
    done
done