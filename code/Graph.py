import copy
import random

useSTK = True

if useSTK:
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
        self.trueAccessIntervals = {}   # indexed in same order as agent_list
        self.noisyAccessIntervals = {}   # indexed in same order as agent_list

    def initialize(self,agent_list):
        for agent_idx in agent_list:
            self.trueAccessIntervals[agent_idx] = []
            self.noisyAccessIntervals[agent_idx] = []

        for agent_idx in agent_list:
            access = self.stk_ref.GetAccessToObject(agent_list[agent_idx].stk_ref)
            accessIntervals = access.ComputedAccessIntervalTimes

            trueAccessInterval = []
            noisyAccessInterval = []

            for i in range(accessIntervals.Count):
                times = accessIntervals.GetInterval(i)
                rand1 = random.gauss(5, 5)
                rand2 = random.gauss(5, 5)
                interval = [times[0], times[1]]
                noisy_interval = [times[0] + rand1, times[1] + rand2]
                # interval = [times[0], times[1]]
                trueAccessInterval.append(interval)
                noisyAccessInterval.append(noisy_interval)

            self.trueAccessIntervals[agent_idx].append(trueAccessInterval)
            self.noisyAccessIntervals[agent_idx].append(noisyAccessInterval)

    def get_visible_agents(self, agent_list, t):
        visible_agents = []
        
        for agent_idx in agent_list:
            if self.in_range(agent_idx,t):
                visible_agents.append(agent_list[agent_idx])

        return visible_agents

    def in_range(self,agent_idx,t):
        for i in range(len(self.trueAccessIntervals[agent_idx])):
            times = self.trueAccessIntervals[agent_idx][i]
            print(times)
            for j in range(len(times)):
                # print(f"Inteval: {self.noisyAccessIntervals[agent_idx][i]}    time:{t}" )
                if t >= times[j][0] and t <= times[j][1]:
                    return True
        return False


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