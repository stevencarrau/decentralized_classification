class Graph:
    def __init__(self):
        self.vertices = {}
        self.agents = []
        self.sensors = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_agents(self, agents):
        self.agents = agents

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def add_sensors(self, sensors):
        self.sensors = sensors

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


class Sensor:
    def __init__(self, name):
        self.name = name
        self.shadow = None

    def add_shadow(self, shadow):
        # TODO: probably take from STK
        pass

    def add_path(self, path):
        # TODO: probably import from STK
        pass

    def __str__(self):
        return str(self.name)


class Agent:
    def __init__(self, name, sensors):
        self.name = name
        self.sensors = sensors

    def __str__(self):
        return str(self.name)


if __name__ == '__main__':
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

    graph = Graph()

    # Scenario 1
    graph.add_sensors([sensor1, sensor2])
    graph.add_agents([agent2, agent3, agent5, agent6, agent7])

    graph.add_vertex(agent2, agent5)
    graph.add_vertex(agent3, agent2)
    graph.add_vertex(agent3, agent5)

    graph.add_vertex(agent6, agent7)

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

    for s in graph.sensors:
        print(s, end=" ")
    print()
    for a in graph.agents:
        print(a, end=" ")
    print("\n\n")
    for v in graph.vertices:
        print(v, end="    ")
        for a in graph.vertices[v]:
            print(a, end=" ")
        print()
