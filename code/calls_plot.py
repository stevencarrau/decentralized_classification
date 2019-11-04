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

with open('5agents_3-3_range_sync_min.json') as json_file:
	data = json.load(json_file)
df = pd.DataFrame(data)
with open('5agents_4-3_range_async_min.json') as json_file2:
	data2 = json.load(json_file2)
df2 = pd.DataFrame(data2)
my_dpi = 100
scale_factor = 1.0
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=2.5, metadata=dict(artist='Me'), bitrate=1800)
fig = texfig.figure(width=3.3*scale_factor,ratio=0.65, dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10",len(df.index))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]
cat2 = [str(d_i) for d_i in df2['0'][0]['Id_no']]
# cat_hold = []
# cat_hold.append(categories.pop())
# cat_hold.append(categories.pop())
# cat_hold.append(categories.pop())
# categories.append(cat_hold.pop(0))
# categories.append(cat_hold.pop(0))
# categories.append(cat_hold.pop(0))
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

line_types = itertools.cycle(["--","-.",":","-"])

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
	rad_obj = update(i)
	grid_obj = grid_update(i)
	conn_obj = connect_update(i)
	belf_obj = belief_update(i)
	return rad_obj + grid_obj+conn_obj + belf_obj

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

def belief_chart_init():
	ax = plt.subplot(111)
	ax.set_xlim([0,frames])
	ax.set_ylim([0,4000])
	ax.yaxis.set_ticks(np.arange(0,4001,1000))
	ax.xaxis.set_ticks(np.arange(0,frames+1,10))
	plt.rc('text',usetex=True)
	plt.xlabel(r't')
	plt.ylabel(r'Cumulative Case One Calls')
	plt_array = []
	j = 1
	belief_calls_A = 0
	belief_calls_B = 0
	belief_x_bad.append([[0],[0]])
	belief_y_bad.append([[0],[0]])
	for i,id_no in enumerate(categories):
		belief_calls_A += df['0'][id_no]['BeliefCalls']
	for i, id_no in enumerate(cat2):
		belief_calls_B += df2['0'][id_no]['BeliefCalls']
		# belief_x_bad[0].append(i)
		# belief_x_bad[1].append(i)
		# belief_y_bad[0].append(belief_calls_A)
		# belief_y_bad[1].append(belief_calls_B)
	px1, = ax.plot([0,0.0], [0,0.0], color='b', linewidth=3*scale_factor, linestyle=next(line_types), label=r'SDHT')
	px2, = ax.plot([0,0.0], [0,0.0], color='r', linewidth=3*scale_factor, linestyle=next(line_types), label=r'ADHT')
	plt_array.append((px1,px2))
	leg = ax.legend(loc='upper left')
	return plt_array

def con_init():
	ax = plt.subplot(111)
	plt.axis('off')
	ax.set_xlim([-ax.get_window_extent().height/2, ax.get_window_extent().height/2])
	ax.set_ylim([-ax.get_window_extent().height/2, ax.get_window_extent().height/2])
	radius = ax.get_window_extent().height/2 - 50
	cir_array = []
	loc_dict = {}
	for col, a_i in enumerate(angles[:-1]):
		loc = tuple(np.array([radius*math.sin(a_i),radius*math.cos(a_i)]))
		loc_dict.update({categories[col]:loc})
		p_c = patches.Circle(loc,36,color=my_palette(col),zorder=4)
		ax.add_artist(p_c)
		cir_array.append(p_c)
	line_dict = {}
	for l_d in loc_dict:
		for k_d in loc_dict:
			x_point,y_point = zip(*[loc_dict[l_d], loc_dict[k_d]])
			p_l, = ax.plot(x_point,y_point,linewidth=5,color=(0,0,0),zorder=1)
			p_l.set_visible(False)
			line_dict.update({(l_d, k_d): p_l})
	return line_dict

def grid_update(i):
	global ax_ar, df, ncols, obs_range
	write_objects = []
	for a_x, id_no in zip(ax_ar, categories):
		c_i, l_i, p_i = a_x
		loc = tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'][0], ncols)))
		c_i.set_center(loc)
		l_i.set_xy(np.array(loc)-obs_range-0.5)
		route_x, route_y = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['NominalTrace'][s][0], ncols))) for s in df[str(i)][str(id_no)]['NominalTrace']])
		p_i.set_xdata(route_x)
		p_i.set_ydata(route_y)
		write_objects += [c_i] + [l_i] + [p_i]
	return write_objects

def belief_update(i):
	global bel_lines, df, belief_x_good, belief_y_good, belief_x_bad, belief_y_bad
	change_array = []
	belief_calls_A = 0
	belief_calls_B = 0
	for j,id_no in enumerate(categories):
		belief_calls_A += df[str(i)][id_no]['BeliefCalls']
	for j, id_no in enumerate(cat2):
		belief_calls_B += df2[str(i)][id_no]['BeliefCalls']
	belief_x_bad[0][0].append(i)
	belief_x_bad[0][1].append(i)
	belief_y_bad[0][0].append(belief_calls_A)
	belief_y_bad[0][1].append(belief_calls_B)
	bel_lines[0][0].set_xdata(belief_x_bad[0][0])
	bel_lines[0][0].set_ydata(belief_y_bad[0][0])
	bel_lines[0][1].set_xdata(belief_x_bad[0][1])
	bel_lines[0][1].set_ydata(belief_y_bad[0][1])
		# bel_lines[j][1].set_xdata(belief_x_bad[j])
		# bel_lines[j][1].set_ydata(belief_y_bad[j])
	change_array += bel_lines
	return change_array

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
moveobstacles = []
obstacles = []
# # # 5 agents small range
# initial = [(33,0),(80,0),(69,1),(7,0),(41,0)]
# targets = [[0,9],[69,95],[99,11],[20,39],[60,69]]
# public_targets = [[0,9],[55,95],[99,11],[20,39],[60,69]]
# obs_range = 4

# # # 6 agents small range
# initial = [(33,0),(41,0),(7,0),(80,0),(69,1),(92,0)]
# targets = [[0,9],[60,69],[20,39],[69,95],[99,11],[9,91]]
# public_targets = [[0,9],[60,69],[20,39],[55,95],[99,11],[9,91]]
# obs_range = 2

# # # 8 agents small range
# initial = [(50,0),(43,0),(75,0),(88,0),(13,0),(37,0),(57,0),(73,0)]
# targets = [[0,90],[3,93],[5,95],[98,8],[11,19],[31,39],[51,59],[55,71]]
# public_targets = [[0,90],[3,93],[5,95],[98,8],[11,19],[31,39],[51,59],[79,71]]
# obs_range = 2

# #4 agents larger range
obs_range = 3

# #4 agents big range
# initial = [(33,0),(41,0),(7,0),(80,0)]
# targets = [[0,9],[60,69],[20,39],[69,95]]
# public_targets = [[0,9],[60,69],[20,39],[55,95]]
# obs_range = 4

# con_dict = con_ar = con_init()'874' (140647750019760) = {dict} <class 'dict'>: {'BeliefCalls': 736, 'BadTrace': {'37': [85, 1], '43': [78, 0], '27': [76, 0], '11': [85, 0], '49': [76, 1], '20': [66, 1], '9': [85, 1], '26': [86, 0], '5': [67, 1], '31': [69, 0], '1': [78, 0], '25': [85, 0], '41': [76, 0], '33': [67, 1], '23': [85, 1], '2': [79, 0], '6': [66, 1], '4': [68, 1], '3': [69, 0], '24': [95, 1], '7': [76, 1], '21': [76, 1], '14': [77, 0], '13': [76, 0], '38': [95, 1], '29': [78, 0], '46': [68, 1], '34': [66, 1], '22': [75, 1], '19': [67, 1], '45': [69, 0], '15': [78, 0], '40': [86, 0], '16': [79, 0], '39': [85, 0], '47': [67, 1], '12': [86, 0], '10': [95, 1], '30': [79, 0], '8': [75, 1], '18': [68, 1], '17': [69, 0], '48': [66, 1], '35': [76, 1], '28': [77, 0], '0': [77, 0], '44': [79, 0], '32': [68, 1], '50': [75, 1], '42': [77, 0], '36': [75, 1]}, 'LastSeen': {'37': [[67, 0], 0], '874': [[77, 0], 0], '122': [[34, 0], 0], '684': [[33, 0], 50], '168': [[27, 0], 8]}, 'ActBelief': {'(0, 0, 1, 0, 0)': 0.0042983371692197755, '(1, 1, 1, 0, 1)': ...… View
time_i = 50
bel_lines = belief_chart_init()
for j in range(time_i+1):
	belief_update(j)
# ax_ar = grid_init(nrows, ncols, obs_range)
# update_all(50)
texfig.savefig("Call_results_"+str(time_i))
# update()
# plt.show()
# ani = FuncAnimation(fig, update_all, frames=frames, interval=500, blit=True,repeat=False)
# plt.show()
# ani.save('8_agents-3range-wheel.mp4',writer = writer)
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