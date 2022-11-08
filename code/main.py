from Graph import Sensor, Agent, Graph
from Graph_Visual import GraphVisual
import matplotlib.pyplot as plt
import random
import copy

sensor1 = Sensor("s1")
sensor2 = Sensor("s2")
sensor3 = Sensor("s3")
sensor4 = Sensor("s4")
sensor5 = Sensor("s5")
sensor6 = Sensor("s6")
sensors = [sensor1, sensor2, sensor3, sensor4, sensor5, sensor6]

agent1 = Agent("a1", [sensor1, sensor2])
agent2 = Agent("a2", [sensor2, sensor3])
agent3 = Agent("a3", [sensor3, sensor4, sensor5])
agent4 = Agent("a4", [sensor3, sensor4, sensor5])
agent5 = Agent("a5", [sensor1, sensor4])
agent6 = Agent("a6", [sensor3, sensor5, sensor6])
agent7 = Agent("a7", [sensor1, sensor2, sensor3])
agent8 = Agent("a8", [sensor6])
agents = [agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8]

graph = GraphVisual()

# Scenario 1
# graph.add_sensors([sensor1, sensor2])
graph.add_agents([agent2, agent3, agent5, agent6, agent7])

graph.add_vertex(agent2, agent5)
graph.add_vertex(agent3, agent2)
graph.add_vertex(agent3, agent5)

graph.add_vertex(agent6, agent7)

T = 10
graph.draw_graph(4)
plt.ion()
plt.show()
for t in range(T):
    for idx, a in enumerate(graph.agents):
        agent_list = copy.deepcopy(graph.agents)
        agent_list.pop(idx)
        new_vertices = random.sample(agent_list,random.randint(0,len(agent_list)))
        graph.add_vertices(a,new_vertices)
    plt.pause(1)
    graph.update_graph()
plt.ioff()
plt.show()