#!/bin/bash

for i in `seq 1 50`;
do
	# parallel python first_runs_experiment.py ::: 3 ::: 0 ::: 0.1 ::: 0.9 ::: 0.1 ::: true ::: 0.0 ::: 10000
    # Agent, Environment, Gamma, Epsilon, Alpha, lambda, Kappa, Action_in_features, initialization, max_steps, num_bins
    
    parallel python main_experiment.py ::: 3 ::: 3 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: true ::: 0.0 ::: 20000 ::: 5 10 20
    # parallel python main_experiment.py ::: 2 ::: 1 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: false ::: 0.0 ::: 20000 ::: 5 10 20
    
    parallel python main_experiment.py ::: 2 ::: 1 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: false ::: 0.0 ::: 20000 ::: 100
    # parallel python main_experiment.py ::: 2 ::: 1 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: false ::: 0.0 ::: 20000 ::: 5 10 20
    # parallel python main_experiment.py ::: 2 ::: 1 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: false ::: 0.0 ::: 20000 ::: 5 10 20
    # parallel python main_experiment.py ::: 2 ::: 1 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: false ::: 0.0 ::: 20000 ::: 5 10 20
    
    parallel python main_experiment.py ::: 3 ::: 3 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0 ::: false true ::: 5.0 ::: 20000 ::: 5 10 20
    # parallel python main_experiment.py ::: 2 ::: 1 ::: 0.99 ::: 0.0 0.1 ::: 0.1 ::: 0.0 ::: 0 ::: false ::: 5.0 ::: 20000 ::: 100
    # parallel python horsetrack_episodic_experiment.py ::: 2 ::: 0 ::: 0.1 ::: 0.0 ::: 0.0001 0.001 0.01 0.1 0 1 10 100 1000 10000 ::: false ::: 0.0 5.0 ::: 100000
done