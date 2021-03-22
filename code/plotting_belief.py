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



# ---------- PART 1: Globals

with open('5agents_6-HV_range_async_min.json') as json_file:
	data = json.load(json_file)
df = pd.DataFrame(data)
my_dpi = 150
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
# fig = texfig.figure(width=2000/my_dpi,dpi=my_dpi)
fig = plt.figure(figsize=(3600/my_dpi, 2000/my_dpi), dpi=my_dpi)
# fig = plt.figure(figsize=(2000/my_dpi, 1600/my_dpi), dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10",len(df.index))
# seed_iter = iter(range(0,5))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]

belief_good = df['0'][0]['GoodBelief']
belief_bad = df['0'][0]['BadBelief']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
axis_array = []
l_data = []
f_data = []
belief_x_good = []
belief_x_bad = []
belief_y_bad = []
belief_y_good = []
frames = 100


def update_all(i):
	grid_obj = grid_update(i)
	return grid_obj


def grid_init(nrows, ncols, obs_range):
	# fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
	ax = plt.subplot(111)
	t = 0
	row_labels = range(nrows)
	col_labels = range(ncols)
	plt.xticks(range(ncols), col_labels)
	plt.yticks(range(nrows), row_labels)
	ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
	ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
	ax.set_xlim(-0.5,ncols-0.5)
	ax.set_ylim(-0.5,nrows-0.5)
	ax.invert_yaxis()
	ag_array = []
	plt.grid(which="minor", ls="-", lw=1)
	i = 0
	for id_no in categories:
		p_t = df[str(0)][id_no]['PublicTargets']
		if id_no == '10' or id_no == '11':
			color = (255.0/255.0, 0.0/255.0, 14.0/255.0,1.0)
		else:
			color = (0/255.0, 254.0/255.0, 10.0/255.0,1.0)
		# color = my_palette(i)
		init_loc = tuple(reversed(coords(df[str(0)][id_no]['AgentLoc'], ncols)))
		c_i = plt.Circle(init_loc, 0.45, color=color)
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['NominalTrace'][s][0],ncols))) for s in df[str(t)][str(id_no)]['NominalTrace']])
		cir_ax = ax.add_artist(c_i)
		ag_array.append([cir_ax])
		# for k in p_t:
		# 	s_c = coords(k, ncols)
		# 	ax.fill([s_c[1]+0.4, s_c[1]-0.4, s_c[1]-0.4, s_c[1]+0.4], [s_c[0]-0.4, s_c[0]-0.4, s_c[0]+0.4, s_c[0]+0.4], color=color, alpha=0.9)
		i += 1
	return ag_array

def grid_update(i):
	global ax_ar, df, ncols, obs_range
	write_objects = []
	for a_x, id_no in zip(ax_ar, categories):
		# c_i, l_i, p_i,p_2 = a_x
		c_i = a_x[0]
		loc = tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'], ncols)))
		c_i.set_center(loc)
		# l_i.set_xy(np.array(loc)-obs_range-0.5)
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['NominalTrace'][s][0], ncols))) for s in df[str(i)][str(id_no)]['NominalTrace']])
		# p_i.set_xdata(route_x)
		# p_i.set_ydata(route_y)
		# route_x2, route_y2 = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['BadTrace'][s][0], ncols))) for s in
		#                          df[str(i)][str(id_no)]['BadTrace']])
		# p_2.set_xdata(route_x2)
		# p_2.set_ydata(route_y2)
		write_objects += [c_i]
	return write_objects

def coords(s,ncols):
	return (int(s /ncols), int(s % ncols))



nrows = 20
ncols = 20

# ---------- PART 2:
regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['deterministic']= range(nrows*ncols)
gwg = Gridworld(initial=[0],nrows=nrows,ncols=ncols,regions=regions)
gwg.render()
gwg.draw_state_labels()

moveobstacles = []
obstacles = []

# #4 agents larger range
obs_range = 6

# con_dict = con_ar = con_init()
# bel_lines = belief_chart_init()
ax_ar = grid_init(nrows, ncols, obs_range)

# ani = FuncAnimation(fig, update_all, frames=frames, interval=20, blit=False,repeat=False)
# plt.show()
# ani.save('12_agents.mp4',writer = writer)

#
ani = FuncAnimation(fig, update_all, frames=50, interval=200, blit=True)
plt.show()
