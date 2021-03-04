import json
import matplotlib
matplotlib.use('pgf')
# matplotlib.use('Qt5Agg')
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
fname = '5agents_3-Z_range_'
with open(fname+'async_min.json') as json_file:
# with open('/home/scarr/Downloads/5agents_4-9_range_async_average.json') as json_file:
	data = json.load(json_file)
data2 = data
# with open(fname+'sync_min.json') as json_file:
# # with open('/home/scarr/Downloads/5agents_4-9_range_async_average.json') as json_file:
# 	data2 = json.load(json_file)
# data2['26']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.58
# data2['27']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.57
# data2['28']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.55
# data2['29']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.53
# data2['30']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.52
# data2['31']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.51
# data2['32']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.51
# data2['33']['4']['ActBelief']['(1, 1, 1, 0, 1)'] = 0.50

df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
my_dpi = 480
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
# fig = texfig.figure(width=2000/my_dpi,dpi=my_dpi)
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
# fig = plt.figure(figsize=(2000/my_dpi, 1600/my_dpi), dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10",len(df.index))
seed_iter = iter(range(0,5))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]
categories = []
for k in seed_iter:
	np.random.seed(k)
	categories.append(k)


color_list = [(0,0,1),(0.09,0.4,0.18),(0,1,1),(0.545,0,0),(1,0,0.56)]
belief_good = df['0'][0]['GoodBelief']
belief_bad = df['0'][0]['BadBelief']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
axis_array = []
l_data = []
f_data = []
belief_x_good = []
belief_x_local = []
belief_y_local = []
belief_y_good = []
belief_x_good_s = []
belief_y_good_s = []
# plt.show()
frames = 1565
agent_no = 4

def belief_chart_init():
	ax = plt.subplot(111,frame_on=True)
	plt.tight_layout()
	ax.set_xlim([0,frames//32])
	ax.set_ylim([-0.1,1.2])
	ax.yaxis.set_ticks(np.arange(0,1.1,0.25))
	plt.rc('text',usetex=True)
	# plt.xlabel(r't')
	# plt.ylabel(r'Belief $\left(b_j(\theta)\right)$')
	plt_array = []
	for i,id_no in enumerate([categories[agent_no]]):
		belief_x_local.append([])
		belief_x_local[i].append(0)
		belief_y_local.append([])
		belief_y_local[i].append(df['0'][id_no]['LocalBelief'][belief_good])
		belief_x_good.append([])
		belief_x_good_s.append([])
		belief_x_good[i].append(0)
		belief_x_good_s[i].append(0)
		belief_y_good.append([])
		belief_y_good_s.append([])
		belief_y_good[i].append(df['0'][id_no]['ActBelief'][belief_good])
		belief_y_good_s[i].append(df2['0'][id_no]['ActBelief'][belief_good])
		px1, = ax.plot([0,0.0], [0,0.0], color=color_list[agent_no], linewidth=3, linestyle='solid', label=r'ADHT')
		px3 = None
		# px3, = ax.plot([0,0.0], [0,0.0], color=color_list[agent_no], linewidth=3, linestyle='dashed', label=r'SDHT')
		px2, = ax.plot([0,0.0], [0.0,0.0], color=color_list[agent_no], linewidth=3, linestyle='dotted', label=r'Local belief')
		plt_array.append((px1,px2,px3))
	leg = ax.legend(loc='lower right')
	return plt_array

def belief_update(i):
	global bel_lines, df, belief_x_good, belief_y_good, belief_x_bad, belief_y_bad
	change_array = []
	for j, id_no in enumerate([categories[agent_no]]):
		belief_x_local[j].append(i//32)
		belief_x_good[j].append(i//32)
		belief_x_good_s[j].append(i//32)
		belief_y_good[j].append(df[str(i//32)][id_no]['ActBelief'][belief_good])
		belief_y_good_s[j].append(df2[str(i//32)][id_no]['ActBelief'][belief_good])
		belief_y_local[j].append(df[str(i//32)][id_no]['LocalBelief'][belief_good])
		bel_lines[j][0].set_xdata(belief_x_good[j])
		bel_lines[j][0].set_ydata(belief_y_good[j])
		# bel_lines[j][2].set_xdata(belief_x_good_s[j])
		# bel_lines[j][2].set_ydata(belief_y_good_s[j])
		bel_lines[j][1].set_xdata(belief_x_local[j])
		bel_lines[j][1].set_ydata(belief_y_local[j])
		change_array += [bel_lines[j][0]]
		change_array += [bel_lines[j][1]]
		change_array += [bel_lines[j][2]]
	return change_array


def update_all(i):
	belf_obj = belief_update(i)
	return belf_obj




def coords(s,ncols):
	return (int(s /ncols), int(s % ncols))




# ---------- PART 2:

nrows = 10
ncols = 10
moveobstacles = []
obstacles = []

bel_lines = belief_chart_init()
# ani = FuncAnimation(fig, update_all, frames=frames, interval=5, blit=True,repeat=False)
# plt.show()

ani = FuncAnimation(fig, update_all, frames=frames, interval=5, blit=False,repeat=False)
ani.save('Agent4_Belief_ADHT_new.mp4',writer = writer)
