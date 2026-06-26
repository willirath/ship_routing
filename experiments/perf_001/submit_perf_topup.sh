#!/usr/bin/env bash
# ONE-OFF: top up W=1 perf replicas for the 3 transient-outlier cases (c03,c10,c11).
# Adds 3 fresh replicas per case (submission-id 2) so the slow runs get outvoted
# in the median. Run once. See plans/topup-underrepresented-replicas.md.
set -e

mkdir -p slurm_logs

echo "Submitting perf_topup.job (9 tasks: c03,c10,c11 x r0..2 at W=1, submission-id 2)"
job=$(sbatch --parsable perf_topup.job)
echo "  Job $job"
echo "Done."
