import json
import matplotlib
# matplotlib.use('pgf')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from math import ceil
from math import pi
import math
from ast import literal_eval
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import (DrawingArea,OffsetImage,AnnotationBbox)
import numpy as np
from gridworld import *
# import texfig
import pickle
import csv

fname = 'Hallway_Sim_12_Agents_NoMeet'
with open(fname+'.json') as json_file:
	nomeet_data = json.load(json_file)

fname = 'Hallway_Sim_12_Agents_Meet'
with open(fname+'.json') as json_file:
	meet_data = json.load(json_file)

belief_calls = np.zeros((len(nomeet_data),3))
belief_calls[:,0] = range(len(nomeet_data))
for i in nomeet_data:
	belief_calls[int(i),1] = np.average([nomeet_data[i][j]['BeliefCalls'] for j in nomeet_data[i]])
	belief_calls[int(i), 2] = np.average([meet_data[i][j]['BeliefCalls'] for j in meet_data[i]])

np.savetxt('belief_calls.csv',belief_calls,delimiter=',')