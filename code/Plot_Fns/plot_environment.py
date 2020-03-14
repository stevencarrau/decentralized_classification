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
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)
import numpy as np
from gridworld import *
# import texfig
import pickle

# ---------- PART 1: Globals
fname = '../Fixed_Env_5_Agents_Range'
with open(fname + '.json') as json_file:
	data = json.load(json_file)
with open(fname + '.pickle', 'rb') as env_file:
	gwg = pickle.load(env_file)
df = pd.DataFrame(data)
my_dpi = 150
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
# fig = texfig.figure(width=2000/my_dpi,dpi=my_dpi)
# fig = plt.figure(figsize=(3600/my_dpi, 2000/my_dpi), dpi=my_dpi)
fig = plt.figure(figsize=(2000 / my_dpi, 1600 / my_dpi), dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10", len(df.index) + 5)
seed_iter = iter(range(0, 5))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]

belief_good = df['0'][0]['GoodBelief']
belief_bad = df['0'][0]['BadBelief']
N = len(df[str(0)][-1]['Targets'])
N_a = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
axis_array = []
l_data = []
f_data = []
belief_x_good = []
belief_x_bad = []
belief_y_bad = []
belief_y_good = []
# plt.show()
frames = 250




def grid_init(gwg):
	# fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
	nrows, ncols, obs_range = gwg.nrows, gwg.ncols, gwg.obs_range
	ax = plt.subplot(111)
	t = 0
	row_labels = range(nrows)
	col_labels = range(ncols)
	plt.xticks(range(ncols), "")
	plt.yticks(range(nrows), "")
	ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
	ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
	ax.set_xlim(-0.5, ncols - 0.5)
	ax.set_ylim(-0.5, nrows - 0.5)
	ax.invert_yaxis()
	ag_array = []
	plt.grid(which="minor", ls="-", lw=1)
	i = 0
	# Obstacles
	for o_i in gwg.obstacles:
		o_loc = tuple(reversed(coords(o_i, ncols)))
		obs_ax = ax.add_patch(patches.Rectangle(np.array(o_loc) - 0.5, 1, 1, fill=True,
												color=(1, 0, 0), clip_on=True, alpha=0.2, ls='--', lw=0))

	for id_no in categories:
		# p_t = df[str(0)][id_no]['PublicTargets']
		color = my_palette(i)
		init_loc = tuple(reversed(coords(df[str(0)][id_no]['AgentLoc'], ncols)))
		c_i = plt.Circle(init_loc, 0.45, color=color)
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['NominalTrace'][s][0],ncols))) for s in df[str(t)][str(id_no)]['NominalTrace']])
		cir_ax = ax.add_artist(c_i)
		lin_ax = ax.add_patch(
			patches.Rectangle(np.array(init_loc) - obs_range - 0.5, 2 * obs_range + 1, 2 * obs_range + 1, fill=False,
							  color=color, clip_on=True, alpha=0.5, ls='--', lw=4))
		# plt_ax, = ax.plot(route_x, route_y, color=color, linewidth=5, linestyle='solid')
		plt_ax = None
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['BadTrace'][s][0], ncols))) for s in df[str(t)][str(id_no)]['BadTrace']])
		# plt_ax2, = ax.plot(route_x, route_y, color=color, linewidth=5, linestyle='--',dashes=[3, 8, 2, 5, 1, 6],alpha=0.8)
		plt_ax2 = None
		# ag_array.append([cir_ax, lin_ax, plt_ax,plt_ax2])
		ag_array.append([cir_ax, lin_ax])
		i += 1

	for k in df[str(0)][-1]['Targets']:
		color = my_palette(i)
		s_c = coords(k, ncols)
		ax.fill([s_c[1] + 0.4, s_c[1] - 0.4, s_c[1] - 0.4, s_c[1] + 0.4],
				[s_c[0] - 0.4, s_c[0] - 0.4, s_c[0] + 0.4, s_c[0] + 0.4], color=color, alpha=0.9)
		i += 1

	return ag_array





def grid_update(i):
	global ax_ar, df, ncols, obs_range
	write_objects = []
	for a_x, id_no in zip(ax_ar, categories):
		# c_i, l_i, p_i,p_2 = a_x
		c_i, l_i = a_x
		loc = tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'], ncols)))
		c_i.set_center(loc)
		l_i.set_xy(np.array(loc) - obs_range - 0.5)
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['NominalTrace'][s][0], ncols))) for s in df[str(i)][str(id_no)]['NominalTrace']])
		# p_i.set_xdata(route_x)
		# p_i.set_ydata(route_y)
		# route_x2, route_y2 = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['BadTrace'][s][0], ncols))) for s in
		#                          df[str(i)][str(id_no)]['BadTrace']])
		# p_2.set_xdata(route_x2)
		# p_2.set_ydata(route_y2)
		write_objects += [c_i] + [l_i]  # + [p_i] + [p_2]
	return write_objects




def coords(s, ncols):
	return (int(s / ncols), int(s % ncols))


# ---------- PART 2:

nrows = gwg.nrows
ncols = gwg.ncols

obs_range = gwg.obs_range



ax_ar = grid_init(gwg)
plt.show()

