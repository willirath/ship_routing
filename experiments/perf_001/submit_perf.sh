#!/usr/bin/env bash
# Submit all 5 performance benchmark jobs (one per worker count).
# Each job dispatches 48 tasks (16 cases x 3 replicas) via xargs/srun.
set -e

mkdir -p slurm_logs

for w in 01 02 04 08 16; do
    echo "Submitting perf_w${w}.job"
    job_id=$(sbatch --parsable "perf_w${w}.job")
    echo "  Job $job_id (${w} workers)"
done

echo "Done. 5 perf benchmark jobs queued."
