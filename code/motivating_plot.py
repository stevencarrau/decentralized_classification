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

with open('4agents_3range_tight.json') as json_file:
	data = json.load(json_file)
data[str(0)]['848']['AgentLoc'][0] = 7
data[str(0)]['128']['AgentLoc'][0] = 10
# data[str(0)]['216']['AgentLoc'][0] = 84
data[str(0)]['496']['AgentLoc'][0] = 49
df = pd.DataFrame(data)
my_dpi = 100
scale_factor = 0.5
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
frames = 100
routes = [[[0.75,0.75],[0.75,-0.25],[1.75,-0.25],[1.75,0.75],[1.75,2.25],[1.75,3.25],[1.75,4.25],[1.25,4.25],[0.75,4.25],[0.75,3],[0.75,2],[0.75,0.75],[0.75,-0.25],[1,-0.25]],
          [[4.0,-0.25],[4.25, -0.25], [4.25, 0.75], [3.25, 0.75], [3.25,2 ], [3.25, 3.25],
           [3.75, 3.25], [3.75, 3], [3.75, 2], [3.75, 1], [3.75, -0.25], [4, -0.25]],
		[[1, 8],[1,8.75], [1.25, 8.75], [1.25, 8.25], [1.75, 8.25], [1.75,7], [1.75, 6], [1.75, 4.75],
           [1.25, 4.75], [1.25, 6], [1.25, 7], [1.25, 8]],
		[[5, 5], [4, 5], [3.25, 5],[3.25, 6], [3.75, 6], [3.75, 7.25], [4.75,7.25], [4.75, 8], [4.75, 8.75],
           [4.25, 8.75], [4.25, 7.75], [2.75, 7.75], [2.75, 7],[2.75, 6],[2.75, 5],[3.25, 5]]
          ]
bad_routes = [[[1.25,1],[1.25,2.0],[1.25,3.0],[1.25,3.75],[0.75,3.75],[0.75,2.75],[-0.25,2.75],[-0.25,2.0],[-0.25,1.0],[-0.25,0.25],[0.25,0.25],[0.25,1],[0.25,2.25],[1.25,2.25]],
          [[4.25,1.25],[4.75,1.25],[4.75,0.25],[5.25,0.25],[5.25,1.75],[5.25,3.75],[4.25,3.75],[4.25,2.5],[4.25,1.75],[4.75,1.75],[4.75,1.25]],
		[[0.75,8],[0.75,6.5],[0.75,5.75],[0.25,5.75],[0.25,6],[0.25,7.0],[0.25,8.0],[0.25,8.75],[-0.25,8.75],[-0.25,8.0],[-0.25,7.25],[0.75,7.25]],
		[[5.25,5],[5.25,6.5],[5.25,8.25],[4.75,8.25],[4.75,8.0],[4.75,6.5],[4.75,5.25],[5.25,5.25]]
          ]

g_arrows = [
	[(1.25,-0.25,0.2,0),(1.75,2,0,0.2),(1,4.25,-0.2,0),(0.75,2.5,0,-0.2)],
	[(3.75, 2.5, 0.0, -0.2), (4.1, -0.25, 0.2, 0), (3.25, 2, 0.0, 0.2)],
	[(1.75,7,0,-0.2),(1.25,7,0,0.2)],
	[(4,5,-0.2,0),(3.25,5.5,0,0.2),(3.5,7.75,-0.2,0),(2.75,6.5,0,-0.2)]
]

b_arrows = [
	[(-0.25,1.5,0.0,-0.2),(0.75,1.25,0.0,0.2),(0.75,2.25,0.2,0),(1.25,1.75,0,0.2)],
	[(5.25, 2, 0.0, 0.2), (4.75, 3.75, -0.2, 0), (4.25, 3, 0.0, -0.2)],
	[(0.75,6.5,0,-0.2),(0.25,6.5,0,0.2),(-0.25,8.25,0,-0.2)],
	[(5.25,6.5,0,0.2),(4.75,6.5,0,-0.2)]
]


def update_all(i):
	grid_obj = grid_update(i)
	return grid_obj

def grid_init(nrows, ncols, obs_range):
	ax = plt.subplot(111)
	t = 0
	row_labels = []
	for r_w in range(nrows):
		if r_w % 2 ==0:
			row_labels.append(r_w)
		else:
			row_labels.append(None)
	col_labels = range(ncols)
	plt.xticks(range(ncols), col_labels)
	plt.yticks(range(nrows), row_labels)
	ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
	ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
	ax.set_xlim(-0.5,ncols-0.5)
	ax.set_ylim(-0.5,nrows-0.5)
	ax.invert_yaxis()
	ag_array = []
	plt.grid(which="minor", ls="-", lw=0.5*scale_factor)
	# init_loc = [3,1]
	# lin_ax = ax.add_patch(
	# 	patches.Rectangle(np.array(init_loc) - obs_range - 0.5, 2 * obs_range + 1, 2 * obs_range + 1, fill=True,
	# 	                  color=(1,1,0), clip_on=True, alpha=0.15, ls='--', lw=1 * scale_factor))
	# init_loc = [4, 8]
	# lin_ax2 = ax.add_patch(
	# 	patches.Rectangle(np.array(init_loc) - 2 - 0.5, 2 * 2 + 1, 2 * 2 + 1, fill=True,`
	# 	                  color=(0,0.64,0), clip_on=True, alpha=0.15, ls='--', lw=1 * scale_factor))
	i = 0
	for id_no in categories:
		p_t = df[str(0)][id_no]['PublicTargets']
		color = my_palette(i)
		init_loc = tuple(reversed(coords(df[str(0)][id_no]['AgentLoc'][0], ncols)))
		c_i = plt.Circle(init_loc, 0.45, color=color)
		route_x, route_y = zip(*routes[i])
		# new_route = []
		# r_x_iter = iter(route_x)
		# next(r_x_iter)
		# r_y_iter = iter(route_y)
		# next(r_y_iter)
		# updown = False
		# dx = 0
		# for ind_r,f_x in enumerate(route_x[:-1]):
		# 	f_nx = next(r_x_iter)
		# 	f_ny = next(r_y_iter)
		# 	dy = np.sign(f_nx-f_x)*0.25 #np.sign(f_ny-route_y[ind_r])*0.25 #+ np.sign(f_nx-f_x)*0.25
		# 	if updown and dy != 0:
		# 		if f_nx-f_x == 0:
		# 			new_route.append([route_x[ind_r]+dx,route_y[ind_r]+dy])
		# 			new_route.append([route_x[ind_r],route_y[ind_r]+dy])
		# 		else:
		# 			new_route.append([route_x[ind_r]+dx,route_y[ind_r]+dy])
		# 			new_route.append([route_x[ind_r],route_y[ind_r]+dy])
		# 	else:
		# 		new_route.append([route_x[ind_r]+dx,route_y[ind_r]+dy])
		# 	dx = np.sign(f_ny - route_y[ind_r]) * 0.25  # np.sign(f_nx-f_x)*0.25 #+ np.sign(f_ny-route_y[ind_r])*0.25
		# 	if dx!= 0: updown = True
		# route_x, route_y = zip(*new_route)
		cir_ax = ax.add_artist(c_i)
		ax.annotate(r'$' + str(i) + '$', xy=(init_loc[0] - 0.1 / scale_factor, init_loc[1] + 0.15 / scale_factor),
		            color=(1, 1, 1), zorder=3)
		# if i==3:
		# 	plt_ax, = ax.plot(route_x, route_y, color=color, linewidth=4 * scale_factor, linestyle=':', alpha=0.8)
		# 	r_x_iter = iter(route_x)
		# 	next(r_x_iter)
		# 	r_y_iter = iter(route_y)
		# 	next(r_y_iter)
		# 	for f_x,f_y in zip(route_x[:-1],route_y[:-1]):
		# 		# print(i)
		# 		dx = next(r_x_iter)-f_x
		# 		dy = next(r_y_iter)-f_y
		# 		plt.arrow(f_x+0.8*dx, f_y+0.8*dy, 0.1*dx*scale_factor, 0.1*dy*scale_factor, shape='full', lw=0, length_includes_head=True, head_width=0.6*scale_factor,color=color)
		# else:
		plt_ax,= ax.plot(route_x, route_y, color=color, linewidth=3 * scale_factor, linestyle='solid')
		for p_i in g_arrows[i]:
			plt.arrow(p_i[0], p_i[1], p_i[2], p_i[3], shape='full', lw=0, length_includes_head=True,
			          head_width=.6 * scale_factor, color=color)
		
		# r_x_iter = iter(route_x)
		# next(r_x_iter)
		# r_y_iter = iter(route_y)
		# next(r_y_iter)
		# for f_x,f_y in zip(route_x[:-1],route_y[:-1]):
		# 	# print(i)
		# 	dx = next(r_x_iter)-f_x
		# 	dy = next(r_y_iter)-f_y
		# 	plt.arrow(f_x+0.8*dx, f_y+0.8*dy, 0.1*dx*scale_factor, 0.1*dy*scale_factor, shape='full', lw=0, length_includes_head=True, head_width=0.6*scale_factor,color=color)
		
		lin_ax =None
		plt_ax = None
		# lin_ax = ax.add_patch(patches.Rectangle(np.array(init_loc)-obs_range-0.5, 2*obs_range+1, 2*obs_range+1,fill=False, color=color, clip_on=True, alpha=0.8, ls=':', lw=1.5*scale_factor))
		
		# plt_ax, = ax.plot(route_x, route_y, color=color, linewidth=2*scale_factor, linestyle='solid')
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['BadTrace'][s][0], ncols))) for s in
		#                          df[str(t)][str(id_no)]['BadTrace']])
		route_x, route_y = zip(*bad_routes[i])
		# if i==3:
		# 	plt_ax2,= ax.plot(route_x, route_y, color=color, linewidth=5 * scale_factor, linestyle='solid')
		# 	r_x_iter = iter(route_x)
		# 	next(r_x_iter)
		# 	r_y_iter = iter(route_y)
		# 	next(r_y_iter)
		# 	for f_x,f_y in zip(route_x[:-1],route_y[:-1]):
		# 		# print(i)
		# 		dx = next(r_x_iter)-f_x
		# 		dy = next(r_y_iter)-f_y
		# 		plt.arrow(f_x+0.8*dx, f_y+0.8*dy, 0.1*dx*scale_factor, 0.1*dy*scale_factor, shape='full', lw=0, length_includes_head=True, head_width=.6*scale_factor,color=color)
		# else:
		# plt_ax2, = ax.plot(route_x, route_y, color=color, linewidth=3*scale_factor, linestyle=':',alpha=0.8)
		# for p_i in b_arrows[i]:
		# 	plt.arrow(p_i[0],p_i[1],p_i[2],p_i[3],shape='full', lw=0, length_includes_head=True, head_width=.6*scale_factor,color=color)
		# plt_ax,= ax.plot(route_x, route_y, color=color, linewidth=2 * scale_factor, linestyle='solid')
		# r_x_iter = iter(route_x)
		# next(r_x_iter)
		# r_y_iter = iter(route_y)
		# next(r_y_iter)
		# for f_x,f_y in zip(route_x[:-1],route_y[:-1]):
		# 	# print(i)
		# 	dx = next(r_x_iter)-f_x
		# 	dy = next(r_y_iter)-f_y
		# 	plt.arrow(f_x+0.8*dx, f_y+0.8*dy, 0.1*dx*scale_factor, 0.1*dy*scale_factor, shape='full', lw=0, length_includes_head=True, head_width=.6*scale_factor,color=color)
		#
		ag_array.append([cir_ax, lin_ax, plt_ax])
		# for k,m in zip(p_t,bad_targets[i]):
		# 	s_c = coords(k, ncols)
		# 	ax.fill([s_c[1]+0.4, s_c[1]-0.4, s_c[1]-0.4, s_c[1]+0.4], [s_c[0]-0.4, s_c[0]-0.4, s_c[0]+0.4, s_c[0]+0.4], color=color, alpha=0.9)
		# 	s_d = coords(m, ncols)
		# 	ax.add_patch(patches.Rectangle(np.array(tuple(reversed(s_d)))-0.4,0.8,0.8, color=color, alpha=0.4,fill=False,lw=5*scale_factor))
		# 	ax.scatter(s_d[1],s_d[0],marker='x',alpha=0.4,s=100,color=color,zorder=-3)
		i += 1
	return ag_array

def grid_update(i):
	global ax_ar, df, ncols, obs_range
	write_objects = []
	for a_x, id_no in zip(ax_ar, categories):
		c_i, l_i, p_i,p_2 = a_x
		loc = tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'][0], ncols)))
		c_i.set_center(loc)
		l_i.set_xy(np.array(loc)-obs_range-0.5)
		route_x, route_y = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['NominalTrace'][s][0], ncols))) for s in df[str(i)][str(id_no)]['NominalTrace']])
		p_i.set_xdata(route_x)
		p_i.set_ydata(route_y)
		route_x2, route_y2 = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['BadTrace'][s][0], ncols))) for s in
		                         df[str(i)][str(id_no)]['BadTrace']])
		p_2.set_xdata(route_x2)
		p_2.set_ydata(route_y2)
		write_objects += [c_i] + [l_i] + [p_i] + [p_2]
	return write_objects



def coords(s,ncols):
	return (int(s /ncols), int(s % ncols))



# ---------- PART 2:

nrows = 10
ncols = 6
moveobstacles = []
obstacles = []
bad_targets = [[0,25],[5,28],[31,54],[35,59]]
obs_range = 2
ax_ar = grid_init(nrows, ncols, obs_range)
texfig.savefig("case_environment_good")
