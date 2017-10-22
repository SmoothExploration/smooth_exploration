#!/bin/bash

for i in `seq 1 5`;
do
	parallel python floating_horsetrack_experiment.py ::: 2 ::: 0 0.1 ::: 0.125 0.25 0.5 1 2 ::: 0 0.25 0.5 0.75 0.9 0.99 1 ::: 0 0.1 1 10 100 ::: true false ::: 0.0 5.0 ::: 10000
done