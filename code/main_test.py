from Graph import Sensor, Graph
from Graph_Visual import GraphVisual
import matplotlib.pyplot as plt
import random
import copy
from STK_Agent import Agent, Observation, Observe
import numpy as np


def max_belief(belief):
    bel = 0
    max_bel = (0, 0, 0, 0, 0)
    for b_i in belief:
        if belief[b_i] >= bel:
            bel = belief[b_i]
            max_bel = b_i
    return {max_bel: bel}


# STK runs sims at 30 fps for this scenario
time_step = 0.033  # seconds
num_timesteps_per_second = 1 / time_step  # about 30 timesteps/sec
time_multiplier = 1.5
T = int(60 * time_multiplier)
true_belief = (1, 1, 1, 1, 0)

# create aircraft/sensor objects with corresponding agents
aircraft = []
sensors = []

# grab the coordinates of washington for position computation
washingon_coords = np.array([1115.07, -4843.94, 3983.24])

for craft in range(1, 6):
    data = np.loadtxt("Aircraft{num}.csv".format(num=craft), delimiter=',')
    times = data[0, :]
    x_values = data[1, :]
    y_values = data[2, :]
    z_values = data[3, :]

    aircraft.append(Agent("Aircraft{num}".format(num=craft), None, true_belief, times, x_values, y_values, z_values))

# create graph
subagents = aircraft
subagents_names = [a_i.name for a_i in subagents]
for a_i in subagents:
    a_i.init_sharing_type(subagents_names)
    a_i.init_belief(len(subagents))
subagents[-1].evil = True

# initialize bimodal pdfs and intervals for each agent
for craft_idx, craft in enumerate(aircraft):
    craft.intialize_bimodal_pdf()
    craft.intialize_interval(craft_idx)

graph = GraphVisual()
graph.add_agents(subagents)

ignore_sensors = []
graph.draw_graph(4)
plt.ion()
plt.show()
for t_idx, t in enumerate(range(T)):
    graph.clear()
    # Loop once to update connections
    for idx, a in enumerate(graph.agents):
        agent_list = copy.copy(graph.agents)
        agent_list.pop(idx)

        new_vertices = {}
        for sensor_idx, sensor in enumerate(sensors):
            # FIXME: change if you want to block sensor(s) for the duration of the experiment
            if sensor_idx in ignore_sensors:
                continue
            new_vertices.update(sensor.query(agent_list, t_idx / time_multiplier))
        new_vertices = list(new_vertices)

        graph.add_vertices(a, new_vertices)
        ## TODO: Give as input the "Observation" function - i.e connected agents that are in their "observable" zone
        observations = Observe.get_observations(graph.agents, int(t_idx / time_multiplier))
        a.updateLocalBelief(observations)
    # Loop again to build sharing graphs
    for idx, a in enumerate(graph.agents):
        belief_packet = dict(
            [[v_a.name, v_a.actual_belief] for v_a in graph.vertices[a]])
        a.shareBelief(belief_packet)
    actual_belief = [a.actual_belief[true_belief] for a in graph.agents]
    local_belief_print = [max_belief(a.local_belief) for a in graph.agents]
    actual_belief_print = [max_belief(a.actual_belief) for a in graph.agents]
    print(f"Local: {local_belief_print}")
    print(f"Actual: {actual_belief_print}")
    plt.pause(1)
    graph.update_graph(actual_belief)
    plt.draw_all()

print("Done!")
plt.ioff()
plt.show()