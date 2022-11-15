import copy
from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkobjects import *

class Graph:
    def __init__(self):
        self.vertices = {}
        self.agents = []

    def clear(self):
        self.vertices = {}

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_agents(self, agents):
        self.agents += agents

    # for now we store these as agent -> list of agents it's connected to
    def add_vertex(self, agent1, agent2):
        if agent1 in self.vertices and agent2 not in self.vertices[agent1]:
            self.vertices[agent1].append(agent2)
        elif agent1 not in self.vertices:
            self.vertices[agent1] = [agent2]

        if agent2 in self.vertices and agent1 not in self.vertices[agent2]:
            self.vertices[agent2].append(agent1)
        elif agent2 not in self.vertices:
            self.vertices[agent2] = [agent1]

    def add_vertices(self, agent1, agent_list):
        self.vertices[agent1] = []
        for agent in agent_list:
            self.add_vertex(agent1, agent)

    def remove_vertex(self, agent1, agent2):
        if agent1 in self.vertices and agent2 in self.vertices[agent1]:
            self.vertices[agent1].remove(agent2)

        if agent2 in self.vertices and agent1 in self.vertices[agent2]:
            self.vertices[agent2].remove(agent1)

    def remove_vertices(self, agent1, agent_list):
        for agent in agent_list:
            self.remove(agent1, agent)


class Sensor:
    def __init__(self, name, stk_ref):
        self.name = name
        self.stk_ref = stk_ref

    def get_visible_agents(self, agent_list, t):
        visible_agents = []
        
        for agent_idx, agent in enumerate(agent_list):
           access = self.stk_ref.GetAccessToObject(agent.stk_ref)
           accessIntervals = access.ComputedAccessIntervalTimes
           
           for i in range(0, accessIntervals.Count):
                times = accessIntervals.GetInterval(i)
                if t >= times[0] and t <= times[1]:
                    visible_agents.append(agent)

        return visible_agents


    def query(self, agent_list, t):
        # create dictionary from every agent -> all other visible agents
        visible_agent_dict = {}
        visible_agents = self.get_visible_agents(agent_list, t)
        for idx, agent in enumerate(visible_agents):
            copied_list = []
            
            # copy every other agent except the one you're on
            for other_agent_idx, other_agent in enumerate(visible_agents):
                if other_agent_idx != idx:
                    copied_list.append(other_agent)

            visible_agent_dict[agent] = copied_list
        return visible_agent_dict

    def __str__(self):
        return str(self.name)





if __name__ == '__main__':
    agent1 = Agent("a1")
    agent2 = Agent("a2")
    agent3 = Agent("a3")
    agent4 = Agent("a4")
    agent5 = Agent("a5")
    agent6 = Agent("a6")
    agent7 = Agent("a7")
    agent8 = Agent("a8")
    agents = [agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8]

    sensor1 = Sensor("s1", [agent1, agent2])
    sensor2 = Sensor("s2", [agent3, agent4])
    sensor3 = Sensor("s3", [agent1, agent5])
    sensor4 = Sensor("s4", [agent8, agent3])
    sensor5 = Sensor("s5", [agent7])
    sensor6 = Sensor("s6", [agent2])
    sensors = [sensor1, sensor2, sensor3, sensor4, sensor5, sensor6]

    graph = Graph()

    # Scenario 1
    graph.add_agents([agent2, agent3, agent5, agent6, agent7])

    # for each timestep
    graph.clear()
    for s in sensors:
        edges = s.query()
        for agent in edges:
            graph.add_vertices(agent, edges[agent])

    # Scenario 2:
    # graph.add_sensors(sensors)
    # graph.add_agents(agents)
    #
    # graph.add_vertex(agent1, agent2)
    # graph.add_vertex(agent2, agent3)
    # # no replication
    # graph.add_vertex(agent2, agent3)
    # graph.add_vertex(agent1, agent3)
    # graph.add_vertex(agent4, agent5)
    # graph.add_vertex(agent1, agent6)
    # graph.add_vertex(agent2, agent7)

    print("\n\n")
    for a in graph.agents:
        print(a, end=" ")
    print("\n\n")
    for v in graph.vertices:
        print(v, end="    ")
        for a in graph.vertices[v]:
            print(a, end=" ")
        print()
