import json
from math import ceil
from math import pi
import math
from ast import literal_eval
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import (DrawingArea,OffsetImage,AnnotationBbox)
import numpy as np
from gridworld import *
import pickle
import csv

fname = 'Hallway_Sim_12_Agents_NoMeet'
with open(fname+'.json') as json_file:
	nomeet_data = json.load(json_file)

fname = 'Hallway_Sim_12_Agents_Meet'
with open(fname+'.json') as json_file:
	meet_data = json.load(json_file)

# belief_calls = np.zeros((len(nomeet_data),3))
# belief_calls[:,0] = range(len(nomeet_data))
# for i in nomeet_data:
# 	belief_calls[int(i),1] = np.average([nomeet_data[i][j]['BeliefCalls'] for j in nomeet_data[i]])
# 	belief_calls[int(i), 2] = np.average([meet_data[i][j]['BeliefCalls'] for j in meet_data[i]])
#
# np.savetxt('belief_calls.csv',belief_calls,delimiter=',')
nomeet_trace = np.zeros((len(nomeet_data),12))
nomeet_dict = dict()
meet_trace = np.zeros((len(nomeet_data),12))
meet_dict = dict()

for t in nomeet_data:
	nomeet_dict[t] = dict()
	meet_dict[t] = dict()
	for k in nomeet_data[t]:
		nomeet_trace[int(t),int(k)] = nomeet_data[t][k]['AgentLoc']
		meet_trace[int(t),int(k)] = meet_data[t][k]['AgentLoc']
		nomeet_dict[t][k] = nomeet_data[t][k]['AgentLoc']
		meet_dict[t][k] = meet_data[t][k]['AgentLoc']

np.savetxt('nomeet_trace.csv',nomeet_trace,delimiter=',')
np.savetxt('meet_trace.csv',meet_trace,delimiter=',')
with open('nomeet_trace.json','w') as fp:
	json.dump(nomeet_dict,fp)

with open('meet_trace.json','w') as fp:
	json.dump(meet_dict,fp)