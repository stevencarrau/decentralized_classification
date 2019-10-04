import json
import matplotlib

# matplotlib.use('pgf')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.collections as collections
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
from scipy.spatial import ConvexHull


# ---------- PART 1: Globals

# with open('8agents_3range_wheel.json') as json_file:
# 	data = json.load(json_file)
# df = pd.DataFrame(data)
my_dpi = 100
scale_factor = 1.0
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=2.5, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure()
# fig = texfig.figure(width=3.3*scale_factor,ratio=1, dpi=my_dpi)
my_palette =  plt.cm.get_cmap("tab10",4)
drone_locs = [(2,2),(4,2),(2,8),(7,7)]
drone_routes = [[(2,2),(1.5,0),(0,1),(0,3),(1,5),(2,4),(2,2)],
[(4,2),(5,1),(7,0),(9,2),(7,4),(4,2)],
[(2,8),(3,5.5),(4,5),(5,7),(3,8.5),(2,8)],
[(7,7),(8,6),(5,5),(6,6),(6,7),(6,8),(7,7)]
]

bad_routes = [[(2,2),(3,0),(4,3),(3,4),(2,2)],
              [(4,2),(6,2),(7,3),(6,4),(4,2)],
              [(2,8),(1,5),(0,7),(0.5,9),(1,9.25),(2,8)],
              [(7,7),(8,6),(9,7),(9.25,9.25),(7.5,9.25),(7,7)]
               ]

# categories = [str(d_i) for d_i in df['0'][0]['Id_no']]
# belief_good = df['0'][0]['GoodBelief']
# belief_bad = df['0'][0]['BadBelief']
# N = len(categories)
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]
# axis_array = []
# l_data = []
# f_data = []
# belief_x_good = []
# belief_x_bad = []
# belief_y_bad = []
# belief_y_good = []
# # plt.show()
# frames = 100
#
# for row in range(0, len(df.index)):
# 	ax = plt.subplot(4, N + 1, row + 1 + int(1 + (N + 1) / 2) * int(row / ((N) / 2)), polar=True)
# 	ax.set_theta_offset(pi / 2)
# 	ax.set_theta_direction(-1)
# 	ax.set_ylim(0, 100)
# 	plt.xticks(angles[:-1], range(N), color='grey', size=8)
# 	for col, xtick in enumerate(ax.get_xticklabels()):
# 		xtick.set(color=my_palette(col), fontweight='bold', fontsize=16)
# 	ax.set_rlabel_position(0)
# 	plt.yticks([25, 50, 75, 100], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=7)
# 	# for j_z, a_i in enumerate(angles[:-1]):
# 	# 	da = DrawingArea(20,20,10,10)
# 	# 	p_c = patches.Circle((0,0), radius=12, color=my_palette(j_z), clip_on=False)
# 	# 	da.add_artist(p_c)
# 	# 	ab = AnnotationBbox(da,(0,101))
# 	# 	ax.add_artist(ab)
# 	l, = ax.plot([], [], color=my_palette(row), linewidth=2, linestyle='solid')
# 	l_f, = ax.fill([], [], color=my_palette(row), alpha=0.4)
# 	axis_array.append(ax)
# 	l_data.append(l)
# 	f_data.append(l_f)
# 	ax.spines["bottom"] = ax.spines["inner"]
# plot_data = [l_data, f_data]

#
# def update_all(i):
# 	rad_obj = update(i)
# 	grid_obj = grid_update(i)
# 	conn_obj = connect_update(i)
# 	belf_obj = belief_update(i)
# 	return rad_obj + grid_obj + conn_obj + belf_obj

#
# def update(i):
# 	global plot_data, df
# 	l_d = plot_data[0]
# 	f_d = plot_data[1]
# 	for l, l_f, id_no in zip(l_d, f_d, categories):
# 		values = df[str(i)][id_no]['ActBelief']
# 		cat_range = range(N)
# 		value_dict = dict([[c_r, 0.0] for c_r in cat_range])
# 		for v_d in value_dict.keys():
# 			for k_i in values.keys():
# 				if literal_eval(k_i)[v_d] == 1:
# 					value_dict[v_d] += 100 * values[k_i]
#
# 		val = list(value_dict.values())
# 		val += val[:1]
# 		l.set_data(angles, val)
# 		l_f.set_xy(np.array([angles, val]).T)
# 	# plot_data = [l_d,f_d]
# 	return l_d + f_d
def convexhull(p):
	p = np.array(p)
	hull = ConvexHull(p)
	return p[hull.vertices,:]

def curvline(start,end,rad,t=100,arrows=1,push=0.8):
    #Compute midpoint
    rad = rad/100.
    x1, y1 = start
    x2, y2 = end
    y12 = (y1 + y2) / 2
    dy = (y2 - y1)
    cy = y12 + (rad) * dy
    #Prepare line
    tau = np.linspace(0,1,t)
    xsupport = np.linspace(x1,x2,t)
    ysupport = [(1-i)**2 * y1 + 2*(1-i)*i*cy + (i**2)*y2 for i in tau]
    #Create arrow data
    arset = list(np.linspace(0,1,arrows+2))
    c = zip([xsupport[int(t*a*push)] for a in arset[1:-1]],
                      [ysupport[int(t*a*push)] for a in arset[1:-1]])
    dt = zip([xsupport[int(t*a*push)+1]-xsupport[int(t*a*push)] for a in arset[1:-1]],
                      [ysupport[int(t*a*push)+1]-ysupport[int(t*a*push)] for a in arset[1:-1]])
    arrowpath = zip(c,dt)
    return xsupport, ysupport, arrowpath

def plotcurv(start,end,rad,color,t=100,arrows=1,arwidth=.25,style='-'):
    x, y, c = curvline(start,end,rad,t,arrows)
    plt.plot(x,y,linestyle=style,color=color)
    for d,dt in c:
        plt.arrow(d[0],d[1],dt[0],dt[1], shape='full', lw=0,
                  length_includes_head=False, head_width=arwidth,color=color)
    return c

class UpdateablePatchCollection(collections.PatchCollection):
    def __init__(self,patches,*args,**kwargs):
        self.patches = patches
        collections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


def aircraft(ax,col,dxdy):
	drone_circles = [
		mpatches.Wedge([0, 0], 0.08, 0, 360, width=0.025, color=col),
		mpatches.Wedge([0.26, 0.26], 0.24, 0, 360, width=0.025, color=col),
		mpatches.Wedge([-0.26, -0.26], 0.24, 0, 360,width=0.025,color=col),
		mpatches.Wedge([0.26, -0.26], 0.24, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, 0.26], 0.24, 0, 360,width=0.025,color=col),
		mpatches.Wedge([0.26, 0.26], 0.21, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, -0.26], 0.21, 0, 360, width=0.025, color=col),
		mpatches.Wedge([0.26, -0.26], 0.21, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, 0.26], 0.21, 0, 360, width=0.025,color=col),
		mpatches.Wedge([0.26, 0.26], 0.25, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, -0.26], 0.25, 0, 360, width=0.025,color=col),
		mpatches.Wedge([0.26, -0.26], 0.25, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, 0.26], 0.25, 0, 360, width=0.025,color=col),
		mpatches.Wedge([0.26, 0.26], 0.025, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, -0.26], 0.025, 0, 360, width=0.025,color=col),
		mpatches.Wedge([0.26, -0.26], 0.025, 0, 360, width=0.025,color=col),
		mpatches.Wedge([-0.26, 0.26], 0.025, 0, 360, width=0.025,color=col)]
	for a_i in drone_circles:
		a_i.set_center(a_i.center + np.array(dxdy))

	drone_verts = [
		[[-0.01, 0.27], [0, 0.26]],
		[[-0.01, 0.25], [0, 0.26]],
		[[-0.01, 0.01], [0.0, 0.0]],
		[[0.01, -0.25], [0, 0.26]],
		[[0.01, -0.27], [0, 0.26]],
		[[0.01, -0.01], [0.0, 0.0]],
		[[-0.01, 0.25], [0, -0.26]],
		[[-0.01, 0.27], [0, -0.26]],
		[[-0.01, 0.01], [0.0, 0.0]],
		[[0.01, -0.25], [0, -0.26]],
		[[0.01, -0.27], [0, -0.26]],
		[[0.01, -0.01], [0.0, 0.0]]
	]
	for d_v in drone_verts:
		d_v[0][0] += dxdy[0]
		d_v[0][1] += dxdy[0]
		d_v[1][0] += dxdy[1]
		d_v[1][1] += dxdy[1]
	drone_lines = []
	for d_v in drone_verts:
		l, = ax.plot(d_v[0], d_v[1], color=col, linewidth=4.0, linestyle='solid')
		drone_lines.append(l)
	drone_patch = UpdateablePatchCollection(drone_circles, edgecolors=col, facecolors=col)
	ax.add_collection(drone_patch)


def grid_init(nrows, ncols, obs_range):
	# fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
	ax = plt.subplot(111)
	
	t = 0
	# row_labels = range(nrows)
	# col_labels = range(ncols)
	# plt.xticks(range(ncols), col_labels)
	# plt.yticks(range(nrows), row_labels)
	# ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
	# ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
	ax.set_xlim(-0.75, nrows - 0.25)
	ax.set_ylim(-0.75, ncols - 0.25)
	ax.invert_yaxis()
	# img = plt.imread('UT_Map.jpeg')
	# ax.imshow(img,extent=[-0.5,9.5,-0.5,9.5])
	# ax.axis('tight')
	plt.axis('off')
	ag_array = []
	init_loc = drone_locs[0]
	
	aircraft(ax,my_palette(i),(0,0))
	# lin_ax = ax.add_patch(
	# 	mpatches.Circle(np.array(init_loc),obs_range, fill=True,
	# 	                  color=(1,1,0), clip_on=True, alpha=0.15, ls='--', lw=1 * scale_factor))
	# init_loc = drone_locs[2]
	# lin_ax2 = ax.add_patch(
	# 	mpatches.Circle(np.array(init_loc),comms_range, fill=True,
	# 	                  color=(0,0.64,0), clip_on=True, alpha=0.15, ls='--', lw=1 * scale_factor))
	# plt.grid(which="minor", ls="-", lw=1)
	# for i in range(4):
	# 	color = my_palette(i)
	# 	init_loc = drone_locs[i]
	# 	aircraft(ax, color, init_loc)
		# outline = [mpath.Path.MOVETO]
		# for ind_i,x_i in enumerate(drone_routes[i][:-1]):
		# 	plotcurv(drone_routes[i][ind_i],drone_routes[i][ind_i+1],50,color,200,arrows=1)
		# for ind_j,xj in enumerate(bad_routes[i][:-1]):
		# 	plotcurv(bad_routes[i][ind_j],bad_routes[i][ind_j+1],50,color,200,style="dashed")
		# for r_i in drone_routes[i]:
		# 	outline.append(mpath.Path.CURVE3)
		# outline[-1] = mpath.Path.CLOSEPOLY
		# path1 = mpatches.PathPatch(mpath.Path(drone_routes[i],outline))
		
	return ag_array


# def belief_chart_init():
# 	ax = plt.subplot(222)
# 	ax.set_xlim([0, frames])
# 	ax.set_ylim([-0.1, 1.2])
# 	ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
# 	plt.rc('text', usetex=True)
# 	plt.xlabel(r't')
# 	plt.ylabel(r'Belief $\left(b^a_j(\theta)\right)$')
# 	plt_array = []
# 	for i, id_no in enumerate(categories):
# 		belief_x_bad.append([])
# 		belief_x_bad[i].append(0)
# 		belief_y_bad.append([])
# 		belief_y_bad[i].append(df['0'][id_no]['ActBelief'][belief_bad])
# 		belief_x_good.append([])
# 		belief_x_good[i].append(0)
# 		belief_y_good.append([])
# 		belief_y_good[i].append(df['0'][id_no]['ActBelief'][belief_good])
# 		px1, = ax.plot([0, 0.0], [0, 0.0], color=my_palette(i), linewidth=3, linestyle='solid',
# 		               label=r'Actual belief: $b^a_' + str(i) + r'(\theta^\star)$')
# 		px2, = ax.plot([0, 0.0], [0.0, 0.0], color=my_palette(i), linewidth=3, linestyle='dashed',
# 		               label=r'Incorrect belief $b^a_' + str(i) + r'(\theta_0)$')
# 		plt_array.append((px1, px2))
# 	leg = ax.legend(loc='upper right')
# 	return plt_array


# def con_init():
# 	ax = plt.subplot(224)
# 	plt.axis('off')
# 	ax.set_xlim([-ax.get_window_extent().height / 2, ax.get_window_extent().height / 2])
# 	ax.set_ylim([-ax.get_window_extent().height / 2, ax.get_window_extent().height / 2])
# 	radius = ax.get_window_extent().height / 2 - 50
# 	cir_array = []
# 	loc_dict = {}
# 	for col, a_i in enumerate(angles[:-1]):
# 		loc = tuple(np.array([radius * math.sin(a_i), radius * math.cos(a_i)]))
# 		loc_dict.update({categories[col]: loc})
# 		p_c = patches.Circle(loc, 36, color=my_palette(col), zorder=4)
# 		ax.add_artist(p_c)
# 		cir_array.append(p_c)
# 	line_dict = {}
# 	for l_d in loc_dict:
# 		for k_d in loc_dict:
# 			x_point, y_point = zip(*[loc_dict[l_d], loc_dict[k_d]])
# 			p_l, = ax.plot(x_point, y_point, linewidth=5, color=(0, 0, 0), zorder=1)
# 			p_l.set_visible(False)
# 			line_dict.update({(l_d, k_d): p_l})
# 	return line_dict

#
# def grid_update(i):
# 	global ax_ar, df, ncols, obs_range
# 	write_objects = []
# 	for a_x, id_no in zip(ax_ar, categories):
# 		c_i, l_i, p_i = a_x
# 		loc = tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'][0], ncols)))
# 		c_i.set_center(loc)
# 		l_i.set_xy(np.array(loc) - obs_range - 0.5)
# 		route_x, route_y = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['NominalTrace'][s][0], ncols))) for s in
# 		                         df[str(i)][str(id_no)]['NominalTrace']])
# 		p_i.set_xdata(route_x)
# 		p_i.set_ydata(route_y)
# 		write_objects += [c_i] + [l_i] + [p_i]
# 	return write_objects


# def belief_update(i):
# 	global bel_lines, df, belief_x_good, belief_y_good, belief_x_bad, belief_y_bad
# 	change_array = []
# 	for j, id_no in enumerate(categories):
# 		belief_x_bad[j].append(i)
# 		belief_x_good[j].append(i)
# 		belief_y_good[j].append(df[str(i)][id_no]['ActBelief'][belief_good])
# 		belief_y_bad[j].append(df[str(i)][id_no]['ActBelief'][belief_bad])
# 		bel_lines[j][0].set_xdata(belief_x_good[j])
# 		bel_lines[j][0].set_ydata(belief_y_good[j])
# 		bel_lines[j][1].set_xdata(belief_x_bad[j])
# 		bel_lines[j][1].set_ydata(belief_y_bad[j])
# 		change_array += bel_lines[j]
# 	return change_array


# def connect_update(i):
# 	global con_dict, df
# 	change_array = []
# 	for id_no in categories:
# 		for id_other in categories:
# 			if int(id_other) in df[str(i)][id_no]['Visible']:
# 				if con_dict[(id_no, id_other)]._visible != True:
# 					con_dict[(id_no, id_other)].set(visible=True, zorder=0)
# 					change_array.append(con_dict[(id_no, id_other)])
# 			else:
# 				if con_dict[(id_no, id_other)]._visible == True:
# 					con_dict[(id_no, id_other)].set(visible=False, zorder=0)
# 					change_array.append(con_dict[(id_no, id_other)])
# 	return change_array


def coords(s, ncols):
	return (int(s / ncols), int(s % ncols))


# ---------- PART 2:

nrows = 1
ncols = 1
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
obs_range = 3.0
comms_range = 2

# #4 agents big range
# initial = [(33,0),(41,0),(7,0),(80,0)]
# targets = [[0,9],[60,69],[20,39],[69,95]]
# public_targets = [[0,9],[60,69],[20,39],[55,95]]
# obs_range = 4

# con_dict = con_ar = con_init()
# bel_lines = belief_chart_init()
# belief_update(10)
# belief_update(20)
# belief_update(40)
i=2
ax_ar = grid_init(nrows, ncols, obs_range)
# grid_update(50)
# plt.show()
# texfig.savefig("init_environment",transparent=True)
# update()
# plt.show()
plt.savefig('drone'+str(i)+'.png',transparent=True)
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