# Set up your Python workspace
# Note: The STK Python API used in this lesson is
# only available with STK 12.1.
# If not installed then use pip to install it.
# pip install agi.stk<..ver..>-py3-none-any.whl
# If using an older version of STK then use win32api or Comtypes

from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkobjects import *
from Graph import Sensor,  Graph
from Graph_Visual import GraphVisual
import matplotlib.pyplot as plt
import random
import copy
from STK_Agent import Agent, Observation, Observe
import numpy as np

# When connected to STK via Python, while creating your variable, 
# using the Tab key after periods enables IntelliSense which displays 
# all of the options available off of the current interface. 
# In the next section you will start STK 

# NOTE FOR STK WEB: you can take advantage of STK/SDF SSO by changing 
# your script to connect to an active instance instead of creating a 
# new instance of STK:
# Connect to an an instance of STK12
stk = STKDesktop.AttachToApplication()
# Create a new instance of STK12.
# Optional arguments set the application visible state and the user-control 
# (whether the application remains open after exiting python).
#stk = STKDesktop.StartApplication(visible=True, userControl=True)
#Check your Task Manager to confirm that STK was called 

# Grab a handle on the STK application root.
root = stk.Root
root.UnitPreferences.Item('DateFormat').SetCurrentUnit('EpSec')
root.UnitPreferences.SetCurrentUnit("DistanceUnit", "km")

# Recall that the AGStkObjectRoot object is at the apex of the STK Object Model. 
# The associated IAgStkObjecetRoot interface will provide the methods and properties to load 
# or create new scenarios and aaccess the Object Model Unit preferences. Through app you have 
# a pointer to the IAgUiApplication interface. How will you obtain a pointer to the IAgStkObjectRoot
# interface? According to IAgUiApplication documentation, the stk.GetObjectRoot() property returns 
# a new instance of the root object of the STK Object Model. 

# Check that the root object has been built correctly, check the type()

type(root)
# output will be 
# agi.stk12.stkobjects.AgStkObjectRoot

# Now that you have launched STK via the Python interface, 
# let's see if we can create a new scenario and set the time 
# period via Python. Create a new scenario, analysis period and 
# reset the animation time.

# 1. Create a new scenario.
# The next task is to create a scenario via the NewScenario method 
# of the IAgStkObjectRoot interface. According to the documentation, 
# the NewScenario method expects to be passed a string representing 
# the name of the scenario, but does not return anything.   
#root.CloseScenario()

#root.NewScenario("STK_Scenario")

# "Rewind" the current scenario (DC) for playing
root.Rewind()
root.AnimationOptions = 2  # eAsniOptionStop
root.Mode = 32  # eAniXRealtime
scenario = root.CurrentScenario
scenario.Animation.AnimStepValue = 1  ;   # second
# scenario.Animation.RefreshDelta = 5   # second


# STK runs sims at 30 fps for this scenario
time_step = 0.033  # seconds
num_timesteps_per_second = 1/time_step   # about 30 timesteps/sec
time_multiplier = 1.5
T = int(scenario.StopTime*time_multiplier)    
true_belief = (1,1,1,1,0)
exp_name = 'Experiment1'

# create aircraft/sensor objects with corresponding agents
aircraft = []
sensors = []

# grab the coordinates of washington for position computation
washingon_coords = np.array([1115.07, -4843.94, 3983.24])

for obj in scenario.Children:
    if isinstance(obj, AgAircraft):
        craftPosDP = obj.DataProviders.Item('Cartesian Position').Group.Item('Fixed').Exec(scenario.StartTime, T*900, 1/(num_timesteps_per_second*30))
        times = craftPosDP.DataSets.GetDataSetByName('Time').GetValues()
        x_values = craftPosDP.DataSets.GetDataSetByName('x').GetValues() - washingon_coords[0]
        y_values = craftPosDP.DataSets.GetDataSetByName('y').GetValues() - washingon_coords[1]
        z_values = craftPosDP.DataSets.GetDataSetByName('z').GetValues() - washingon_coords[2]

        aircraft.append(Agent(obj.InstanceName, obj, true_belief, times, x_values, y_values, z_values))

    elif isinstance(obj, AgPlace):
        for sensor in obj.Children:
            sensors.append(Sensor(sensor.InstanceName, sensor))

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
    craft.dumpcsv()


print("[INFO] Done with initialization")

# for craft_idx, craft in enumerate(aircraft):
#      craft.plotinfo()

graph = GraphVisual()
graph.add_agents(subagents)
agent_list = {i:copy.copy(a_i) for i, a_i in  enumerate(graph.agents)}
[s_i.initialize(agent_list) for s_i in sensors]


def max_belief(belief):
    bel = 0
    max_bel = (0, 0, 0, 0, 0)
    for b_i in belief:
        if belief[b_i] >= bel:
            bel = belief[b_i]
            max_bel = b_i
    return {max_bel: bel}

drone_colors = ['yellow','blue','red','white','green']


ignore_sensors = []
# print(sensors)
graph.draw_graph(4)
[a.draw_belief_graph(4,T,drone_colors[a_idx]) for a_idx,a in enumerate(graph.agents)]
plt.ion()
plt.show()
for t_idx, t in enumerate(range(T)):
    graph.clear()
    # Loop once to update connections
    for idx, a in enumerate(graph.agents):
        agent_list = {i:copy.copy(a_i) for i, a_i in  enumerate(graph.agents) if i !=idx}
        
        new_vertices = {}
        for sensor_idx, sensor in enumerate(sensors):
            # FIXME: change if you want to block sensor(s) for the duration of the experiment
            if sensor_idx in ignore_sensors or not sensor.in_range(idx, t_idx/time_multiplier):
                pass
            else:
                new_vertices.update(sensor.query(agent_list, t_idx/time_multiplier))
        new_vertices = list(new_vertices)
        print(new_vertices)
        graph.add_vertices(a, new_vertices)
        ## TODO: Give as input the "Observation" function - i.e connected agents that are in their "observable" zone
        observations = Observe.get_observations(graph.agents, sensors, int(t_idx/time_multiplier))
        a.updateLocalBelief(observations)
    # Loop again to build sharing graphs
    for idx,a in enumerate(graph.agents):
        belief_packet = dict(
            [[v_a.name, v_a.actual_belief] for v_a in graph.vertices[a]])
        a.shareBelief(belief_packet)
    local_belief = [a.local_belief[true_belief] for a in graph.agents]
    actual_belief = [a.actual_belief[true_belief] for a in graph.agents]
    # print(f"Local: {local_belief}")
    # print(f"Actual: {actual_belief}")
    local_belief_print = [max_belief(a.local_belief) for a in graph.agents]
    actual_belief_print = [max_belief(a.actual_belief) for a in graph.agents]
    print(f"Local: {local_belief_print}")
    print(f"Actual: {actual_belief_print}")
    plt.pause(1)
    graph.update_graph(actual_belief)
    graph.save_graph(exp_name,t_idx)
    [a.update_graph(t_idx, a.actual_belief[true_belief]) for a in graph.agents]
    [a.save_belief(exp_name, a_idx, t_idx) for a_idx, a in enumerate(graph.agents)]
    plt.draw_all()
    for _ in range(int(num_timesteps_per_second/time_multiplier)):
        root.StepForward()

print("Done!")
plt.ioff()
plt.show()