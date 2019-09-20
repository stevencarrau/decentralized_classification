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
import itertools
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import (DrawingArea,OffsetImage,AnnotationBbox)
import numpy as np
from gridworld import *
import texfig



# ---------- PART 1: Globals

with open('4agents_3range_tight.json') as json_file:
	data = json.load(json_file)
df = pd.DataFrame(data)
my_dpi = 100
scale_factor = 0.33
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=2.5, metadata=dict(artist='Me'), bitrate=1800)
fig = texfig.figure(width=3.3*scale_factor,ratio=1, dpi=my_dpi)
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
frames = 50

# marker = itertools.cycle(('p', 'd', 'x', '*'))

# for row in range(0, len(df.index)):
# 	ax = plt.subplot(4, N+1, row+1+int(1+(N+1)/2)*int(row/((N)/2)), polar=True)
# 	ax.set_theta_offset(pi/2)
# 	ax.set_theta_direction(-1)
# 	ax.set_ylim(0,100)
# 	plt.xticks(angles[:-1], range(N), color='grey', size=8)
# 	for col,xtick in enumerate(ax.get_xticklabels()):
# 		xtick.set(color=my_palette(col),fontweight='bold',fontsize=16)
# 	ax.set_rlabel_position(0)
# 	plt.yticks([25, 50, 75, 100], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=7)
# 	# for j_z, a_i in enumerate(angles[:-1]):
# 	# 	da = DrawingArea(20,20,10,10)
# 	# 	p_c = patches.Circle((0,0), radius=12, color=my_palette(j_z), clip_on=False)
# 	# 	da.add_artist(p_c)
# 	# 	ab = AnnotationBbox(da,(0,101))
# 	# 	ax.add_artist(ab)
# 	l, = ax.plot([],[],color=my_palette(row),linewidth=2,linestyle='solid')
# 	l_f, = ax.fill([],[],color=my_palette(row),alpha=0.4)
# 	axis_array.append(ax)
# 	l_data.append(l)
# 	f_data.append(l_f)
# 	ax.spines["bottom"] = ax.spines["inner"]
# plot_data = [l_data, f_data]

def update_all(i):
	conn_obj = connect_update(i)
	return conn_obj




def con_init():
	ax = plt.subplot(111)
	plt.axis('off')
	ax.set_xlim([-ax.get_window_extent().height/2, ax.get_window_extent().height/2])
	ax.set_ylim([-ax.get_window_extent().height/2, ax.get_window_extent().height/2])
	radius = ax.get_window_extent().height/2 - 20*scale_factor
	cir_array = []
	loc_dict = {}
	for col, a_i in enumerate(angles[:-1]):
		loc = tuple(np.array([radius*math.sin(a_i),radius*math.cos(a_i)]))
		loc_dict.update({categories[col]:loc})
		p_c = patches.Circle(loc,12*scale_factor,color=my_palette(col),zorder=4)
		ax.add_artist(p_c)
		cir_array.append(p_c)
	line_dict = {}
	for l_d in loc_dict:
		for k_d in loc_dict:
			x_point,y_point = zip(*[loc_dict[l_d], loc_dict[k_d]])
			p_l, = ax.plot(x_point,y_point,linewidth=3*scale_factor,color=(0,0,0),zorder=1)
			p_l.set_visible(False)
			line_dict.update({(l_d, k_d): p_l})
	return line_dict

def connect_update(i):
	global con_dict, df
	change_array = []
	for id_no in categories:
		for id_other in categories:
			if int(id_other) in df[str(i)][id_no]['Visible']:
				if con_dict[(id_no,id_other)]._visible != True:
					con_dict[(id_no, id_other)].set(visible=True, zorder=0)
					change_array.append(con_dict[(id_no, id_other)])
			else:
				if con_dict[(id_no,id_other)]._visible == True:
					con_dict[(id_no, id_other)].set(visible=False, zorder=0)
					change_array.append(con_dict[(id_no, id_other)])
	return change_array
	


def coords(s,ncols):
	return (int(s /ncols), int(s % ncols))



# ---------- PART 2:

nrows = 10
ncols = 10
obs_range = 3
con_dict = con_ar = con_init()

t_i = 40
for s_i in range(0,t_i):
	update_all(s_i)
	texfig.savefig("images/Connections"+str(s_i))