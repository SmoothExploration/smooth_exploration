from flask import Flask, render_template, request
import numpy as np
import sys
import os
import json
# from pprint import pprint

app = Flask(__name__)

# states = "data/save_runs/states/sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat.npy"
# q_values = "data/save_runs/q_values/sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat.npy"
# feature_counts = "data/save_runs/feature_counts/sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat.npy"
# actions = "data/save_runs/actions/sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat.npy"
# results = "data/sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat"

# print(os.listdir())

# func_q = "data_func_1/save_runs/q_values/sarsa_func__floating_horsetrack__0.0__0.1__0.0__0.0001__False__0.0.dat.npy"
# func_features = "data_func_1/save_runs/feature_counts/sarsa_func__floating_horsetrack__0.0__0.1__0.0__0.0001__False__0.0.dat.npy"
# func_q_data = np.load(func_q)
# func_feature_data = np.load(func_features)
# func_r = "data_func_1/save_runs/rewards/sarsa_func__floating_horsetrack__0.0__0.1__0.0__0.0001__False__0.0.dat.npy"
# func_reward_data = np.load(func_r)
# print(func_reward_data[0])
# print(func_feature_data.shape)
# print(func_q_data[0][1])


# state_data = np.load(states)
# q_data = np.load(q_values)
# feature_data = np.load(feature_counts)
# action_data = np.load(actions)
# results_data = np.loadtxt(results)
# pprint(feature_counts)


@app.route('/')
def index():
    files = [file for file in os.listdir("data") if file[:5] == 'sarsa']
    return render_template('index.html', files=files)

@app.route('/func')
def func():
    files = [file for file in os.listdir("data_func_1") if file[:5] == 'sarsa']
    return render_template('func.html', files=files)

@app.route('/func_data', methods=['GET', 'POST'])
def func_data():
    filename = request.json['run']
    states = "data_func_1/save_runs/states/{}.npy".format(filename)
    q_values = "data_func_1/save_runs/q_values/{}.npy".format(filename)
    feature_counts = "data_func_1/save_runs/feature_counts/{}.npy".format(filename)
    actions = "data_func_1/save_runs/actions/{}.npy".format(filename)
    state_data = np.load(states)
    # state_data = (state_data + 50.0) % 100
    state_data = state_data.tolist()
    q_data = np.load(q_values).tolist()
    feature_data = np.load(feature_counts).tolist()
    action_data = np.load(actions).tolist()

    data = {"states" : state_data,
            "q_values" : q_data,
            "features" : feature_data,
            "actions" : action_data,}

    print(q_data[0])

    return json.dumps(data)

@app.route('/data', methods=['GET', 'POST'])
def data():
    filename = request.json['run']
    # filename = "sarsa_tabular__horsetrack__0.0__0.125__0.0__1.0__False__5.0.dat"
    states = "data/save_runs/states/{}.npy".format(filename)
    q_values = "data/save_runs/q_values/{}.npy".format(filename)
    feature_counts = "data/save_runs/feature_counts/{}.npy".format(filename)
    actions = "data/save_runs/actions/{}.npy".format(filename)
    state_data = np.load(states)
    # state_data = (state_data + 50) % 100
    state_data = state_data.tolist()
    q_data = np.load(q_values).tolist()
    feature_data = np.load(feature_counts).tolist()
    action_data = np.load(actions).tolist()
    
    data = {"states" : state_data,
            "q_values" : q_data,
            "features" : feature_data,
            "actions" : action_data,}

    return json.dumps(data)
