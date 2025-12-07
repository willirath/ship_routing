#!/usr/bin/env bash
## for n in {1..10}; do 
## 	for mm in 01 03 05 07 09 11; do
## 		sbatch --export=TIME_START="2021-${mm}-01T00:00:00",JOURNEY_NAME="Atlantic" run_experiment.job;
## 	done
## done
for n in {1..10}; do 
	for mm in 01 03 05 07 09 11; do
		for spd in 8.0 12.0; do
		       	sbatch --export=TIME_START="2021-${mm}-01T00:00:00",SPEED_KNOTS="${spd}",JOURNEY_NAME="Atlantic" run_experiment.job;
		done
	done
done
