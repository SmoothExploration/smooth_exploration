#!/bin/bash

for i in `seq 1 1`;
do
	# parallel python first_runs_experiment.py ::: 3 ::: 0 ::: 0.1 ::: 0.9 ::: 0.1 ::: true ::: 0.0 ::: 10000
    parallel python horsetrack_episodic_experiment.py ::: 2 ::: 0 ::: 0.1 ::: 0.0 ::: 0.000001 0.00001 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 100000 1000000 ::: false ::: 0.0 5.0 ::: 100000
done