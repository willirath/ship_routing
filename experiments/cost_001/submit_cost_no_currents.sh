#!/usr/bin/env bash
# Submit two independent no-currents ablation submissions.
set -e

mkdir -p slurm_logs

echo "Submitting no-currents ablation: submission 1"
job1=$(sbatch --parsable --export=ALL,SUBMISSION_ID=1 run_cost_no_currents.job)
echo "  Job $job1"

echo "Submitting no-currents ablation: submission 2"
job2=$(sbatch --parsable --export=ALL,SUBMISSION_ID=2 run_cost_no_currents.job)
echo "  Job $job2"

echo "Done. Two independent no-currents submissions queued."
