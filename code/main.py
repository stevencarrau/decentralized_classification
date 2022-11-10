from Graph import Sensor,  Graph
from Graph_Visual import GraphVisual
import matplotlib.pyplot as plt
import random
import copy
from STK_Agent import Agent
import numpy as np

agent1 = Agent("a1")
agent2 = Agent("a2")
agent3 = Agent("a3")
agent4 = Agent("a4")
agent5 = Agent("a5")
agent6 = Agent("a6")
agent7 = Agent("a7")
agent8 = Agent("a8")
agents = [agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8]
subagents = [agent2, agent3, agent5, agent6, agent7]
subagents_names = [a_i.name for a_i in subagents]
for a_i in subagents:
    a_i.init_sharing_type(subagents_names)
    a_i.init_belief(len(subagents))
subagents[-1].evil = True

sensor1 = Sensor("s1", [agent1, agent2])
sensor2 = Sensor("s2", [agent3, agent4])
sensor3 = Sensor("s3", [agent1, agent5])
sensor4 = Sensor("s4", [agent8, agent3])
sensor5 = Sensor("s5", [agent7])
sensor6 = Sensor("s6", [agent2])
sensors = [sensor1, sensor2, sensor3, sensor4, sensor5, sensor6]

graph = GraphVisual()

# Scenario 1
# graph.add_sensors([sensor1, sensor2])
graph.add_agents(subagents)

graph.add_vertex(agent2, agent5)
graph.add_vertex(agent3, agent2)
graph.add_vertex(agent3, agent5)

graph.add_vertex(agent6, agent7)

T = 25
true_belief = (1,1,1,1,0)
graph.draw_graph(4)
plt.ion()
plt.show()
for t in range(T):
    graph.clear()
    # Loop once to update connections
    for idx, a in enumerate(graph.agents):
        agent_list = copy.copy(graph.agents)
        agent_list.pop(idx)
        new_vertices = random.sample(agent_list,random.randint(0,len(agent_list)-2))
        graph.add_vertices(a,new_vertices)
        a.updateLocalBelief()
    # Loop again to build sharing graphs
    for idx,a in enumerate(graph.agents):
        belief_packet = dict(
            [[v_a.name, v_a.actual_belief] for v_a in graph.vertices[a]])
        a.shareBelief(belief_packet)
    local_belief = [a.local_belief[true_belief] for a in graph.agents]
    actual_belief = [a.actual_belief[true_belief] for a in graph.agents]
    print(f"Local: {local_belief}")
    print(f"Actual: {actual_belief}")
    plt.pause(1)
    graph.update_graph(actual_belief)
    plt.draw_all()
plt.ioff()
plt.show()