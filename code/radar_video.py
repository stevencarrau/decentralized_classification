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
# with open('/home/scarr/Downloads/5agents_4-9_range_async_average.json') as json_file:
	data = json.load(json_file)
df = pd.DataFrame(data)
my_dpi = 150
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
# fig = texfig.figure(width=2000/my_dpi,dpi=my_dpi)
fig = plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
# fig = plt.figure(figsize=(2000/my_dpi, 1600/my_dpi), dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10",len(df.index))
seed_iter = iter(range(0,5))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]
categories = []
for k in seed_iter:
	np.random.seed(k)
	categories.append(k)

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
frames = 1565
agent_no = 4

ax = plt.subplot(1,1,1, polar=True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
ax.set_ylim(0,100)
plt.xticks(angles[:-1], range(N), color='grey', size=8)
color_list = [(0,0,1),(0.09,0.4,0.18),(0,1,1),(0.545,0,0),(1,0,0.56)]
for col,xtick in enumerate(ax.get_xticklabels()):
	xtick.set(color=color_list[col],fontweight='bold',fontsize=16)
ax.set_rlabel_position(0)
ax.tick_params(pad=-2.0)
ax.set_rlabel_position(45)
plt.yticks([50, 100], ["", ""], color="grey", size=7)
	# for j_z, a_i in enumerate(angles[:-1]):
	# 	da = DrawingArea(20,20,10,10)
	# 	p_c = patches.Circle((0,0), radius=12, color=my_palette(j_z), clip_on=False)
	# 	da.add_artist(p_c)
	# 	ab = AnnotationBbox(da,(0,101))
	# 	ax.add_artist(ab)
l, = ax.plot([],[],color=color_list[agent_no],linewidth=2,linestyle='solid')
l_f, = ax.fill([],[],color=color_list[agent_no],alpha=0.4)
axis_array.append(ax)
l_data.append(l)
f_data.append(l_f)
ax.spines["bottom"] = ax.spines["inner"]
plot_data = [l_data, f_data]

def update_all(i):
	rad_obj = update(i)
	return rad_obj

def update(i):
	global plot_data, df
	l_d = plot_data[0]
	f_d = plot_data[1]
	for l,l_f,id_no in zip(l_d,f_d,[categories[agent_no]]):
		values = df[str(i//32)][id_no]['ActBelief']
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
moveobstacles = []
obstacles = []

obs_range = 4


# ani = FuncAnimation(fig, update_all, frames=frames, interval=500, blit=False,repeat=False)
# ani.save('Agent4_radar_new.mp4',writer = writer)
ani = FuncAnimation(fig, update_all, frames=frames, interval=50, blit=True,repeat=False)
plt.show()
# ani.save('decen.gif',dpi=80,writer='imagemagick')


# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='polar')
# ax.set_ylim(0,100)
#
# data = np.random.rand(50)*6+2
# theta = np.linspace(0,2.*np.pi, num=50)
# l,  = ax.plot([],[])
#
# def update(i):
#     global data
#     data += (np.random.rand(50)+np.cos(i*2.*np.pi/50.))*2
#     data[-1] = data[0]
#     l.set_data(theta, data )
#     return l,
#
# ani = FuncAnimation(fig, update, frames=50, interval=200, blit=True)
# plt.show()