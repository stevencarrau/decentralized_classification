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
import texfig



# ---------- PART 1: Globals

with open('5agents_4range_async.json') as json_file:
	data = json.load(json_file)
df = pd.DataFrame(data)
my_dpi = 100
scale_factor = 0.33
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=2.5, metadata=dict(artist='Me'), bitrate=1800)
fig = texfig.figure(width=3.3*scale_factor,ratio=2, dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10",len(df.index))
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
# plt.show()
frames = 100

for row in range(0, 1):
	ax = plt.subplot(1, 1, row+1, polar=True)
	ax.set_theta_offset(pi/2)
	ax.set_theta_direction(-1)
	ax.set_ylim(0,100)
	plt.xticks(angles[:-1], range(N), color='grey')
	for col,xtick in enumerate(ax.get_xticklabels()):
		xtick.set(color=my_palette(col),fontweight='bold')
	ax.tick_params(pad=-7.0)
	ax.set_rlabel_position(45)
	# plt.yticks([50,100], ["0.5", "1"], color="grey")
	plt.yticks([50,100], [], color="grey")
	# for j_z, a_i in enumerate(angles[:-1]):
	# 	da = DrawingArea(20,20,10,10)
	# 	p_c = patches.Circle((0,0), radius=12, color=my_palette(j_z), clip_on=False)
	# 	da.add_artist(p_c)
	# 	ab = AnnotationBbox(da,(0,101))
	# 	ax.add_artist(ab)
	l, = ax.plot([],[],color=(0.99, 0.99, 0.59),linewidth=2*scale_factor,linestyle='solid')
	l_f, = ax.fill([],[],color=(0.99, 0.99, 0.59),alpha=0.4)
	axis_array.append(ax)
	l_data.append(l)
	f_data.append(l_f)
	ax.spines["bottom"] = ax.spines["inner"]
plot_data = [l_data, f_data]

def update_all(i):
	rad_obj = update(i)
	# grid_obj = grid_update(i)
	# conn_obj = connect_update(i)
	# belf_obj = belief_update(i)
	return rad_obj

def update(i):
	global plot_data, df
	l_d = plot_data[0]
	f_d = plot_data[1]
	for l,l_f,id_no in zip(l_d,f_d,categories):
		values = df[str(i)][id_no]['ActBelief']
		cat_range = range(N)
		value_dict = dict([[c_r, 0.0] for c_r in cat_range])
		for v_d in value_dict.keys():
			for k_i in values.keys():
				if literal_eval(k_i)[v_d] == 1:
					value_dict[v_d] += 100 * values[k_i]

		val = list(value_dict.values())
		val += val[:1]
		l.set_data(angles,val)
		l_f.set_xy(np.array([angles,val]).T)
	# plot_data = [l_d,f_d]
	return l_d + f_d


	


def coords(s,ncols):
	return (int(s /ncols), int(s % ncols))



# ---------- PART 2:

nrows = 10
ncols = 10
obs_range = 3
update_all(1)
texfig.savefig("images/initial_belief")
# t_i = 40
# for s_i in range(1,50):
# 	update_all(s_i)
# 	texfig.savefig("images/Belief_simplex_async"+str(s_i))
