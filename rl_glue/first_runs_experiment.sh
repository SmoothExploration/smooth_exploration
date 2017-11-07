#!/bin/bash

for i in `seq 1 1`;
do
	parallel python first_runs_experiment.py ::: 3 ::: 0 ::: 0.1 ::: 0.9 ::: 0.1 ::: true ::: 0.0 ::: 10000
done