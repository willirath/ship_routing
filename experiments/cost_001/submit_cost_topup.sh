#!/usr/bin/env bash
# ONE-OFF: balance per-speed replica coverage up to the 12 kn level (~650).
# Tops up 8, 16, 20 kn (12 kn is already the ceiling), baseline + no-currents.
# Per-speed submission counts (yield ~110/submission, ~75 for failure-prone 8 kn):
#   8 kn -> 3,  16 kn -> 2,  20 kn -> 4   = 9 submissions x 2 forcings = 18 jobs.
# Each speed-batch gets a globally unique submission ID (3..11), reused across the
# two forcings (separate file namespaces). Run once. If a speed lands short after
# re-running the analysis, submit one more with the next unused ID.
# See plans/topup-underrepresented-replicas.md.
set -e

mkdir -p slurm_logs

SPEEDS=(8 16 20)
NSUBS=(3 2 4)

sid=3
for idx in "${!SPEEDS[@]}"; do
    SPEED=${SPEEDS[$idx]}
    N=${NSUBS[$idx]}
    for ((b = 0; b < N; b++)); do
        for FORCING in baseline no_currents; do
            if [ "$FORCING" = "no_currents" ]; then EXTRA=",NO_CURRENTS=1"; else EXTRA=""; fi
            echo "Submitting ${SPEED}kn ${FORCING} (submission id ${sid})"
            job=$(sbatch --parsable --export=ALL,SUBMISSION_ID=${sid},TOPUP_SPEED=${SPEED}${EXTRA} run_cost_topup.job)
            echo "  Job $job"
        done
        sid=$((sid + 1))
    done
done

echo "Done. Queued $(((${NSUBS[0]} + ${NSUBS[1]} + ${NSUBS[2]}) * 2)) top-up submissions (8/16/20 kn x baseline+no-currents)."
