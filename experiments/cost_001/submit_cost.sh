#!/usr/bin/env bash
# Submit two independent cost analysis submissions.
set -e

mkdir -p slurm_logs

echo "Submitting cost run: submission 1"
job1=$(sbatch --parsable --export=ALL,SUBMISSION_ID=1 run_cost.job)
echo "  Job $job1"

echo "Submitting cost run: submission 2"
job2=$(sbatch --parsable --export=ALL,SUBMISSION_ID=2 run_cost.job)
echo "  Job $job2"

echo "Done. Two independent submissions queued."
