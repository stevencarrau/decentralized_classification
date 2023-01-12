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
from STK_Agent import Agent
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

root.Rewind()
root.AnimationOptions = 2  # eAsniOptionStop
root.Mode = 32  # eAniXRealtime
scenario = root.CurrentScenario
scenario.Animation.AnimStepValue = 1  ;   # second
# scenario.Animation.RefreshDelta = 5   # second


# create aircraft/sensor objects with corresponding agents
aircraft = []
sensors = []

for obj in scenario.Children:
    if isinstance(obj, AgAircraft):
        aircraft.append(Agent(obj.InstanceName, obj))
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

graph = GraphVisual()
graph.add_agents(subagents)

# seems hard to change
time_step = 0.033  # seconds
num_timesteps_per_second = 1/time_step   # about 30 timesteps/sec
time_multiplier = 1.5
T = int(scenario.StopTime*time_multiplier)

true_belief = (1,1,1,1,0)
graph.draw_graph(4)
plt.ion()
plt.show()
for t in range(T):
    # print("\n\n\n\n")
    # print(t)
    # print("\n\n\n\n")
    graph.clear()
    # Loop once to update connections
    for idx, a in enumerate(graph.agents):
        agent_list = copy.copy(graph.agents)
        agent_list.pop(idx)
        
        new_vertices = {}
        for sensor in sensors:
            new_vertices.update(sensor.query(agent_list, t/time_multiplier))
        new_vertices = list(new_vertices)
        
        graph.add_vertices(a, new_vertices)
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
    for _ in range(int(num_timesteps_per_second/time_multiplier)):
        root.StepForward()

print("Done!")
plt.ioff()
plt.show()