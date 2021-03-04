import json
import matplotlib

# matplotlib.use('pgf')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
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
import csv

# ---------- PART 1: Globals
fname = '12agents_6-HV_range_async_min'
with open(fname + '.json') as json_file:
	data = json.load(json_file)
with open(fname + '.pickle', 'rb') as env_file:
	gwg = pickle.load(env_file)
df = pd.DataFrame(data)
my_dpi = 200
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
# fig = texfig.figure(width=2000/my_dpi,dpi=my_dpi)
# fig = plt.figure(figsize=(3600/my_dpi, 2000/my_dpi), dpi=my_dpi)
fig = plt.figure(figsize=(1600 / my_dpi, 900 / my_dpi), dpi=my_dpi)
# fig.set_size_inches(w,h)
my_palette = plt.cm.get_cmap("tab10", 10)
seed_iter = iter(range(0, len(df.index)))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]
# categories = []
# for k in seed_iter:
# 	np.random.seed(k)
# 	categories.append(str(np.random.randint(1000)))

belief_good = df['0'][0]['GoodBelief']
belief_bad = df['0'][0]['BadBelief']
# N = len(df[str(0)][-1]['Targets'])
# N_a = len(categories)
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]
axis_array = []
l_data = []
f_data = []
belief_x_good = []
belief_x_bad = []
belief_y_bad = []
belief_y_good = []
# plt.show()
frames = 100

# for row in range(0, N_a):
# 	ax = plt.subplot(4, N_a + 1, row + 1 + int(1 + (N_a + 1) / 2) * int(row / ((N_a) / 2)), polar=True)
# 	ax.set_theta_offset(pi / 2)
# 	ax.set_theta_direction(-1)
# 	ax.set_ylim(0, 100)
# 	plt.xticks(angles[:-1], range(N), color='grey', size=8)
# 	for col, xtick in enumerate(ax.get_xticklabels()):
# 		xtick.set(color=my_palette(len(categories) + col), fontweight='bold', fontsize=16)
# 	ax.set_rlabel_position(0)
# 	ax.tick_params(pad=-5.0)
# 	ax.set_rlabel_position(45)
# 	plt.yticks([50, 100], ["0.50", "1.00"], color="grey", size=7)
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


def update_all(i):
	# rad_obj = update(i)
	grid_obj = grid_update(i)
	# conn_obj = connect_update(i)
	# belf_obj = belief_update(i)
	return grid_obj


def write_csv(filename, data):
	with open(filename, 'w') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in data.items():
			list_val = [value[v_i]['ActBelief']['(0, 1, 1)'] for v_i in value]
			writer.writerow([key, *list_val])


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
# 	plot_data = [l_d, f_d]
# 	return l_d + f_d


def grid_init(gwg):
	plt.tight_layout()
	# fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
	nrows, ncols, obs_range = gwg.nrows, gwg.ncols, gwg.obs_range
	ax = plt.subplot(111)
	ax.set_facecolor((155.0/255.0, 155.0/255.0, 155.0/255.0))
	[i.set_linewidth(5) for i in ax.spines.values()]
	t = 0
	row_labels = range(nrows)
	col_labels = range(ncols)

	plt.xticks([])
	plt.yticks([])
	# plt.xticks("", "")
	# plt.yticks("", "")
	# ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
	# ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
	ax.set_xlim(-0.5, ncols - 0.5)
	ax.set_ylim(-1, nrows)
	ax.invert_yaxis()
	ag_array = []
	# plt.grid(which="minor", ls="-", lw=1)
	i = 0
	# Obstacles
	for o_i in gwg.obstacles:
		o_loc = tuple(reversed(coords(o_i, ncols)))
		obs_ax = ax.add_patch(patches.Rectangle(np.array(o_loc) - 0.5, 1, 1, fill=True,
												color='white', clip_on=True, alpha=1.0, ls='--', lw=0))

	for id_no in categories:
		# p_t = df[str(0)][id_no]['PublicTargets']
		if id_no == '3':
			color = (255.0/255.0, 0.0/255.0, 144.0/255.0,1.0)
		elif id_no =='10':
			color = (255.0/255.0, 134/255.0, 00/255.0,1.0)
		else:
			color = (158.0/255.0, 37.0/255.0, 16.0/255.0,1.0)
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
	#
	# for k in df[str(0)][-1]['Targets']:
	# 	if k==24:
	# 		color = (8.0/255.0, 158.0/255.0,95.0/255.0,1.0)
	# 	else:
	# 		color = (8.0/255.0, 59.0/255.0,158.0/255.0,1.0)
	# 	s_c = coords(k, ncols)
	# 	ax.fill([s_c[1] + 0.4, s_c[1] - 0.4, s_c[1] - 0.4, s_c[1] + 0.4],
	# 			[s_c[0] - 0.4, s_c[0] - 0.4, s_c[0] + 0.4, s_c[0] + 0.4], color=color, alpha=0.9)
	# 	i += 1

	# for m_s in gwg.meeting_states:
	# 	color = (1.0, 0.875, 0)
	# 	s_c = coords(m_s, ncols)
	# 	ax.fill([s_c[1] + 0.4, s_c[1] - 0.4, s_c[1] - 0.4, s_c[1] + 0.4],
	# 			[s_c[0] - 0.4, s_c[0] - 0.4, s_c[0] + 0.4, s_c[0] + 0.4], color=color, alpha=0.95)

	return ag_array


def grid_update(i):
	global ax_ar, df, ncols, obs_range
	write_objects = []
	for a_x, id_no in zip(ax_ar, categories):
		# c_i, l_i, p_i,p_2 = a_x
		c_i, l_i = a_x
		loc = tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'], ncols)))
		c_i.set_center(loc)
		l_i.set_xy(np.array(loc)-obs_range-0.5)
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['NominalTrace'][s][0], ncols))) for s in df[str(i)][str(id_no)]['NominalTrace']])
		# p_i.set_xdata(route_x)
		# p_i.set_ydata(route_y)
		# route_x2, route_y2 = zip(*[tuple(reversed(coords(df[str(i)][str(id_no)]['BadTrace'][s][0], ncols))) for s in
		#                          df[str(i)][str(id_no)]['BadTrace']])
		# p_2.set_xdata(route_x2)
		# p_2.set_ydata(route_y2)
		write_objects += [c_i] + [l_i] # + [p_i] + [p_2]
	return write_objects

# def belief_chart_init():
# 	ax = plt.subplot(111, frame_on=True)
# 	plt.subplots_adjust(bottom=0.1)
# 	plt.subplots_adjust(left=0.1)
# 	# ax1= ax.twinx()
# 	# ax2= ax.twiny()
# 	# plt.tight_layout()
# 	# for axis in ['top', 'bottom', 'left', 'right']:
# 	# 	ax1.spines[axis].set_linewidth(6)
# 	# 	ax1.spines[axis].set_color('black')
# 	# 	ax2.spines[axis].set_linewidth(6)
# 	# 	ax2.spines[axis].set_color('black')
# 	# ax1.spines['top'].set_visible(False)
# 	# ax1.spines['right'].set_visible(False)
# 	# ax2.spines['top'].set_visible(False)
# 	# ax2.spines['right'].set_visible(False)
# 	# ax1.spines['left'].set_position(('axes',0.0))
# 	# ax2['bottom'].set_position(('axes', 0.0))
# 	#
# 	# ax1.set_xlim([0, frames])
# 	# ax1.set_ylim([0.0, 1.1])
# 	# ax2.set_xlim([0, frames])
# 	# ax2.set_ylim([0.0, 1.1])
# 	# ax1.xaxis.set_ticks([20,40,60,80,100])
# 	# ax1.set_xticklabels(["","","","",""])
# 	# ax1.yaxis.set_ticks([0.25,0.5,0.75,1.0])
# 	# ax2.xaxis.set_ticks([20,40,60,80,100])
# 	# ax2.set_xticklabels(["","","","",""])
# 	# ax2.yaxis.set_ticks([0.25,0.5,0.75,1.0])
# 	# plt.rc('text', usetex=True)
# 	# # plt.xlabel(r't')
# 	# # ax.tick_params(axis='x',bottom=False,top=True,labelbottom=False)
# 	# ax1.tick_params(axis="x", direction="in", pad=-15,length=18,width=7,labelcolor='black',colors='black')
# 	# ax1.tick_params(axis='y',direction='in', pad=-100,length=18,width=7,labelsize=36,labelcolor='black',colors='black')
# 	# ax2.tick_params(axis="x", direction="in", pad=-15,length=18,width=7,labelcolor='black',colors='black')
# 	# ax2.tick_params(axis='y',direction='in', pad=-100,length=18,width=7,labelsize=36,labelcolor='black',colors='black')
#
# 	# ax = plt.subplot(111, frame_on=True)
# 	# plt.tight_layout()
# 	plt.plot([0,500],[0,0],transform=ax.get_xaxis_transform(),linewidth=12,color='black',clip_on=False)
# 	plt.plot([0,0],[0,1.1],transform=ax.get_yaxis_transform(),linewidth=12,color='black',clip_on=False)
# 	# lc = LineCollection(np.array([[0,100],[0,0]],colors=['black'],linewidths=6,transform=ax.get_xaxis_transform())
# 	for axis in ['top', 'bottom', 'left', 'right']:
# 		ax.spines[axis].set_linewidth(5)
# 		ax.spines[axis].set_color('yellow')
# 	ax.spines['top'].set_visible(False)
# 	ax.spines['right'].set_visible(False)
# 	ax.set_xlim([0, frames])
# 	ax.set_ylim([0, 1.1])
# 	xticks_loc= [100,200,300,400,500]
# 	ax.xaxis.set_ticks(xticks_loc)
# 	ax.set_xticklabels(["","","","",""])
# 	for x_t in xticks_loc:
# 		plt.plot([x_t,x_t],[0,0.05],linewidth=12,color='black',clip_on=False,zorder=0)
# 	# plt.plot([0,0],[0,0],transform=ax.get_yaxis_transform(),linewidth=8,color='black',clip_on=False)
# 	yticks_loc = [0.25,0.5,0.75,1.0]
# 	ax.yaxis.set_ticks(yticks_loc)
# 	ax.set_yticklabels(["","","",""])
# 	for y_t in yticks_loc:
# 		plt.plot([0,3],[y_t,y_t],linewidth=12,color='black',clip_on=False,zorder=0)
# 	# plt.rc('text', usetex=True)
# 	# plt.xlabel(r't')
# 	# ax.tick_params(axis='x',bottom=False,top=True,labelbottom=False)
# 	ax.tick_params(axis="x", direction="in", pad=-15,length=15,width=5,labelcolor='yellow',colors='yellow')
# 	ax.tick_params(axis='y',direction='in', pad=-100,length=15,width=5,labelsize=36,labelcolor='yellow',colors='yellow')
# 	# plt.text(5, 1.0, '1.00', fontsize=33, color='black')
# 	t_p = []
# 	for y_i in yticks_loc:
# 		tmp = plt.text(5,y_i,'{:.2f}'.format(y_i),fontsize=32,color='yellow')
# 		tmp.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='black')])
# 		t_p.append(tmp)
# 	# plt.ylabel(r'Belief $\left(b^a_j(\theta)\right)$')
# 	plt_array = []
# 	for i, id_no in enumerate([categories[10]]):
# 		belief_x_bad.append([])
# 		belief_x_bad[i].append(0)
# 		belief_y_bad.append([])
# 		belief_y_bad[i].append(df['0'][id_no]['ActBelief'][belief_bad])
# 		belief_x_good.append([])
# 		belief_x_good[i].append(0)
# 		belief_y_good.append([])
# 		belief_y_good[i].append(df['0'][id_no]['ActBelief'][belief_good])
# 		# px2, = ax.plot([0,0.0], [0.0,0.0], color=my_palette(i), linewidth=3, linestyle='dashed', label=r'Incorrect belief $b^a_'+str(i)+r'(\theta_0)$')
# 		px2, = ax.plot([0, 0.0], [0, 0.0], color='black', linewidth=15, linestyle='solid',
# 					   label=r'$b^a_' + str(i) + r'(\theta^\star)$')
# 		px1, = ax.plot([0, 0.0], [0, 0.0], color=my_palette(i), linewidth=12, linestyle='solid',
# 					   label=r'$b^a_' + str(i) + r'(\theta^\star)$')
# 		plt_array.append((px1, px2))
# 	# leg = ax.legend(loc='right')
# 	# leg = ax.legend(bbox_to_anchor=(1.25, 0.85))
# 	return plt_array


# def con_init():
# 	ax = plt.subplot(224)
# 	plt.axis('off')
# 	ax.set_xlim([-ax.get_window_extent().height / 2, ax.get_window_extent().height / 2])
# 	ax.set_ylim([-ax.get_window_extent().height / 2, ax.get_window_extent().height / 2])
# 	radius = ax.get_window_extent().height / 2 - 50
# 	cir_array = []
# 	loc_dict = {}
# 	con_angles = [n / float(N_a) * 2 * pi for n in range(N_a)]
# 	con_angles += con_angles[:1]
# 	for col, a_i in enumerate(con_angles[:-1]):
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




# def belief_update(i):
# 	global bel_lines, df, belief_x_good, belief_y_good, belief_x_bad, belief_y_bad
# 	change_array = []
# 	for j, id_no in enumerate([categories[10]]):
# 		# belief_x_bad[j].append(i)
# 		belief_x_good[j].append(i)
# 		belief_y_good[j].append(df[str(i)][id_no]['ActBelief'][belief_good])
# 		# belief_y_bad[j].append(df[str(i)][id_no]['ActBelief'][belief_bad])
# 		bel_lines[j][0].set_xdata(belief_x_good[j])
# 		bel_lines[j][0].set_ydata(belief_y_good[j])
# 		bel_lines[j][1].set_xdata(belief_x_good[j])
# 		bel_lines[j][1].set_ydata(belief_y_good[j])
# 		change_array += [bel_lines[j][0]]
# 	# change_array += [bel_lines[j][1]]
# 	return change_array


def coords(s, ncols):
	return (int(s / ncols), int(s % ncols))


# ---------- PART 2:

nrows = gwg.nrows
ncols = gwg.ncols

# #4 agents larger range
obs_range = gwg.obs_range
# obs_range = 0


# con_dict = con_ar = con_init()
ax_ar = grid_init(gwg)
for k_i in range(100):
	grid_update(k_i)
	# plt.savefig('data/Hallways/NoMeet/Minimap/minimaps_{0:03d}.png'.format(k_i), transparent=True)
	plt.savefig('data/Minimap/minimaps_{0:03d}.png'.format(k_i),bbox_inches='tight')
# ax_ar = grid_init(gwg)
# write_csv('Meeting_Belief.csv',data)
# update_all(50)
# texfig.savefig("test")
# update()
# plt.show()
# for i in range(10):

