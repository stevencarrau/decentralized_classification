import json
import matplotlib

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
import itertools
from darpa_model import ERSA_Env,track_outs

event_names = {0: 'nominal', 1: 'iceA', 2: 'iceB', 3: 'iceC', 4: 'alarmA', 5: 'alarmB', 6: 'alarmG'}
mc_dict = dict()
env_states = [0, 1, 2, 3, 4, 5, 6, 7]
mdp_list = ERSA_Env()
for a in event_names.keys():
	mc_dict.update({a:[m.construct_MC(dict([[s,[a]] for s in env_states])) for m in mdp_list]})

init_states = [0,1,2,3,4,5]
mc_dict = dict()
for a in event_names.keys():
	mc_dict.update({a:[m.construct_MC(dict([[s,[a]] for s in env_states])) for m in mdp_list]})



def belief_update(belief,likelihood):
	new_belief = np.multiply(belief,likelihood)
	new_belief = new_belief/np.sum(new_belief)
	return new_belief

class Agent():
	def __init__(self, c_i, label,bad_i,mdp,state, t_i, belief=0):
		self.label = label
		self.c_i = c_i
		self.b_i = bad_i
		self.t_i = t_i
		self.belief_values = np.ones((len(init_states),1))/len(init_states)
		self.belief = 0  # All agents presumed innocent to begin with
		self.state = state
		self.mdp = mdp
		self.track_queue = []

	def likelihood(self,a,next_s,mc_dict):
		return np.array([m_i[(self.state, next_s)] for m_i in mc_dict[a]]).reshape((-1, 1))

	def update_value(self,a,next_s):
		self.belief_values = belief_update(self.belief_values,self.likelihood(a,next_s,mc_dict))

	## Belief update rule for each agent
	def update_belief(self,belief,bad_idx):
		self.belief = belief[bad_idx][0]
		if self.belief_line:
			val = [75 * b_i[0] + 25 for b_i in belief]
			val += val[:1]
			angles = [n / float(len(belief)) * 2 * pi for n in range(len(belief))]
			angles += angles[:1]
			self.belief_line.set_data(angles, val)
			self.belief_line.set_zorder(3)
			self.belief_fill.set_xy(np.array([angles, val]).T)
			self.belief_line.set_zorder(2)
			self.belief_artist.set_zorder(10)
			if self.belief > 0.75:
				self.belief_line.set_color('red')
				self.belief_fill.set_color('red')
			return [self.belief_line, self.belief_fill, self.belief_artist]
		return None

	def init_belief_plt(self, l_i, l_f, l_a):
		self.belief_line = l_i
		self.belief_fill = l_f
		self.belief_artist = l_a


class Simulation():
	def __init__(self, ani):
		self.ani = ani
		self.ani.running = True
		self.ani.event = 0
		self.state = None  # TODO: maybe use this later?

# ---------- PART 1: Globals
## Fast convergence
# with open('AgentPaths_MDP_Fast.json') as json_file:
# 	data = json.load(json_file)
# Original model -- lots of nominal paths
# with open('AgentPaths_MDP.json') as json_file:
# 	data = json.load(json_file)
#
# df = pd.DataFrame(data)
my_dpi = 150
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure(figsize=(3600 / my_dpi, 2000 / my_dpi), dpi=my_dpi)
categories = range(6) #[str(d_i) for d_i in df['0'][0]['Id_no']]

belief_y_good = []
frames = 500

# image stuff
triggers = ["nominal", "ice_cream_truck", "fire_alarm", "explosion"]
trigger_image_paths = 3*['pictures/ice_cream.png'] + 2*['pictures/fire_alarm.png']
trigger_image_xy = [(8,19),(15,19),(22,19),(4.5,12),(13,27)]

agent_image_paths = ['pictures/captain_america.png', 'pictures/black_widow.png', 'pictures/hulk.png',
			   'pictures/thor.png', 'pictures/thanos.png', 'pictures/ironman.png']
agents = []
simulation = None

def on_press(event):
	global simulation
	ani = simulation.ani

	if event.key.isspace():
		if ani.running:
			ani.event_source.stop()
		else:
			ani.event_source.start()
		ani.running ^= True
	# TODO: trigger change in states, this is just proof of concept
	elif event.key.lower() == "j":  # trigger ice cream 1
		ani.event = 0
	elif event.key.lower() == "1":  # trigger ice cream 1
		ani.event = 1
	elif event.key.lower() == "2":  # trigger ice cream 2
		ani.event = 2
	elif event.key.lower() == "3":  # trigger ice cream 3
		ani.event = 3
	elif event.key.lower() == "z":  # trigger alarm A
		ani.event = 4
	elif event.key.lower() == "x":  # trigger alarm B
		ani.event = 5
	elif event.key.lower() == "c":  # trigger both alarms A and B
		ani.event = 6

	# update simulation animation
	simulation.ani = ani

def update_all(i):
	grid_obj = grid_update(i)
	return grid_obj


def grid_init(nrows, ncols):
	global agents
	# fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
	ax = plt.subplot2grid((len(categories), 5), (0, 1), rowspan=len(categories), colspan=4)
	t = 0

	row_labels = range(nrows)
	col_labels = range(ncols)
	plt.xticks(range(ncols), '')
	plt.yticks(range(nrows), '')
	ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
	ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
	ax.set_xlim(-0.5, ncols - 0.5)
	ax.set_ylim(-0.5, nrows - 0.5)
	ax.invert_yaxis()
	ag_array = []
	tr_array = []
	plt.grid(which="minor", ls="-", lw=1)
	i = 0

	# define a few colors
	brown = (255.0 / 255.0, 179.0 / 255.0, 102.0 / 255.0)
	gray = (211.0 / 255.0, 211.0 / 255.0, 211.0 / 255.0)
	blue = (173.0 / 255.0, 216.0 / 255.0, 230.0 / 255.0)
	mustard = (255.0 / 255.0, 225.0 / 255.0, 77.0 / 255.0)

	# dark_blue = (0.0 / 255.0, 0.0 / 255.0, 201.0 / 255.0)
	# dark_green = (0.0 / 255.0, 102.0 / 255.0, 0.0 / 255.0)
	# orange = (230.0 / 255.0, 115.0 / 255.0, 0.0 / 255.0)
	# violet = (170.0 / 255.0, 0.0 / 255.0, 179.0 / 255.0)
	# lavender = (255.0 / 255.0, 123.0 / 255.0, 251.0 / 255.0)
	# colors = [orange, violet, dark_green, lavender, mustard, dark_blue]

	# bad ppl: thanos (threat MDP)
	# good ppl: Captain A (Store A MDP), Iron man (home MDP), black widow (store B MDP),
	# Hulk (repairman MDP), Thor (shopper MDP)
	names = ["Store A Owner", "Store B owner", "Repairman", "Shopper", "Suspicious", "Home Owner"]

	storeA_squares = [list(range(m,m+5)) for m in range(366,486,30)]
	home_squares = [list(range(m,m+5)) for m in range(380,500,30)]
	storeB_squares = [list(range(m,m+5)) for m in range(823,890,30)]
	building_squares = list(itertools.chain(*home_squares))+list(itertools.chain(*storeA_squares))+list(itertools.chain(*storeB_squares))
	building_doors = [458,472,825]

	for idx, id_no in enumerate(categories):
		# p_t = df[str(0)][id_no]['PublicTargets']
		# color = colors[int(id_no)]
		# color = my_palette(i)
		samp_out = mdp_list[idx].sample(init_states[idx],0)
		track_init = track_outs((init_states[idx],samp_out))
		init_loc = tuple(reversed(coords(track_init[0]-30, ncols)))
		# c_i = plt.Circle(init_loc, 0.45, label=names[int(id_no)], color=color)
		t_i = plt.text(x=init_loc[0],y=init_loc[1],s=names[idx], fontsize='xx-small')
		c_i = AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no)]), zoom=0.13),
							 xy=init_loc, frameon=False)
		c_i.set_label(names[idx])
		b_i = plt.Circle([init_loc[0]+1,init_loc[1]-1], 0.25, label=names[int(id_no)], color='r')
		# b_i.set_visible(False)

		currAgent = Agent(c_i=c_i, label=names[idx], bad_i=b_i, mdp=mdp_list[idx], state=init_states[idx], t_i=t_i)
		agents.append(currAgent)

		t_i = None

		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['NominalTrace'][s][0],ncols))) for s in df[str(t)][str(id_no)]['NominalTrace']])
		cir_ax = ax.add_artist(currAgent.c_i)
		bad_ax = ax.add_artist(currAgent.b_i)
		ag_array.append([cir_ax,bad_ax])

	for t_p,t_l in zip(trigger_image_paths,trigger_image_xy):
		t_i = AnnotationBbox(OffsetImage(plt.imread(t_p), zoom=0.04),
						 xy=t_l, frameon=False)
		# t_i.set_label(trigger_image_paths[trigger_image_path_index])
		trigger_ax = ax.add_artist(t_i)
		tr_array.append([trigger_ax])
		# legend = plt.legend(handles=cir_ax, loc=4, fontsize='small', fancybox=True)

	for h_s in building_squares:
		h_loc = tuple(reversed(coords(h_s, ncols)))
		ax.fill([h_loc[0]-0.5,h_loc[0]+0.5,h_loc[0]+0.5,h_loc[0]-0.5],[h_loc[1]-0.5,h_loc[1]-0.5,h_loc[1]+0.5,h_loc[1]+0.5],color=brown, alpha=0.8)

	for b_d in building_doors:
		b_loc = tuple(reversed(coords(b_d, ncols)))
		ax.fill([b_loc[0] - 0.5, b_loc[0] + 0.5, b_loc[0] + 0.5, b_loc[0] - 0.5],
				[b_loc[1] - 0.5, b_loc[1] - 0.5, b_loc[1] + 0.5, b_loc[1] + 0.5], color=mustard, alpha=0.8)

	# Fence
	ax.fill([5 + 0.5, 9 + 0.5, 16 + 0.5, 16 + 0.5, 5 + 0.5],
			[12 - 0.5, 3 + 0.5, 3 + 0.5, 12 - 0.5, 12 - 0.5], color=gray, alpha=0.9)

	# Electric Utility Control Boxbad_i
	ax.fill([12 + 0.5, 14 + 0.5, 14 + 0.5, 12 + 0.5],
			[5 + 0.5, 5 + 0.5, 7 + 0.5, 7 + 0.5], color=blue, alpha=0.9)

	# Street
	ax.fill([-1.0 + 0.5, 29.5 + 0.5, 29.5 + 0.5, -1.0 + 0.5],
			[17 + 0.5, 17 + 0.5, 24 + 0.5, 24 + 0.5], color=gray, alpha=0.9)



	## Plot for belief charts
	ax_list = []
	angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
	angles += angles[:1]
	for idx, id_no in enumerate(categories):
		ax_list.append(plt.subplot(len(categories), 5, idx * 5 + 1, polar=True))
		ax_list[-1].set_theta_offset(pi / 2)
		ax_list[-1].set_theta_direction(-1)
		ax_list[-1].set_ylim(0, 100)
		plt.xticks(angles[:-1], "", color='grey', size=8)
		# for col, xtick in enumerate(ax_list[-1].get_xticklabels()):
		# 	xtick.set(color=my_palette(col), fontweight='bold', fontsize=16)
		ax_list[-1].set_rlabel_position(0)
		ax_list[-1].tick_params(pad=-7.0)
		ax_list[-1].set_rlabel_position(45)
		plt.yticks([50, 100], ["", ""], color="grey", size=7)
		# for j_z, a_i in enumerate(angles[:-1]):
		# 	da = DrawingArea(20,20,10,10)
		# 	p_c = patches.Circle((0,0), radius=12, color=my_palette(j_z), clip_on=False)
		# 	da.add_artist(p_c)
		# 	ab = AnnotationBbox(da,(0,101))
		# 	ax.add_artist(ab)
		l, = ax_list[-1].plot([], [], color='green', linewidth=2, linestyle='solid')
		l_f, = ax_list[-1].fill([], [], color='green', alpha=0.4)
		ax_list[-1].spines["bottom"] = ax_list[-1].spines["inner"]
		l_a = ax_list[-1].add_artist(
			AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no)]), zoom=0.08), xy=(0, 0),
						   frameon=False))
		agents[idx].init_belief_plt(l, l_f, l_a)


	return agents,tr_array,building_squares


def grid_update(i):
	global tr_ar, df, ncols, obs_range,building_squares, agents, simulation
	write_objects = []

	if not simulation.ani.running:
		return write_objects

	active_event = simulation.ani.event
	if active_event == 0:
		for t_i in tr_ar:
			t_i[0].set_visible(False)
			write_objects += t_i
	else:
		for ind,t_i in enumerate(tr_ar):
			if active_event == ind+1:
				t_i[0].set_visible(True)
				write_objects += t_i
			else:
				t_i[0].set_visible(False)
				write_objects += t_i
		if active_event == 6:
			tr_ar[-1][0].set_visible(True)
			tr_ar[-2][0].set_visible(True)
			write_objects += [tr_ar[-2][0],tr_ar[-1][0]]


	for agent_idx, agent in enumerate(agents):
		if len(agent.track_queue) ==0:
			next_s = agent.mdp.sample(agent.state,simulation.ani.event)
			agent.track_queue += track_outs((agent.state,next_s))
			agent.update_value(simulation.ani.event,next_s)
			agent.state = next_s
		c_i = agent.c_i
		b_i = agent.b_i
		text_i = agent.t_i
		agent_pos = agent.track_queue.pop(0)
		loc = tuple(reversed(coords(agent_pos-30, ncols)))
		# Use below line if you're working with circles
		b_i.set_center([loc[0]+1,loc[1]-1])

		# Use this line if you're working with images
		c_i.xy = loc
		c_i.xyann = loc
		c_i.xybox = loc

		# update text positions
		text_i.set_visible(False)   # remove old label
		text_i.set_position((loc[0]-1.5, loc[1]+1.5))  # move label to new (x, y)
		text_i.set_visible(True)
		agent.t_i = text_i

		if agent_pos in building_squares:
			c_i.offsetbox.image.set_alpha(0.35)
		else:
			c_i.offsetbox.image.set_alpha(1.0)

		agent.c_i = c_i
		write_objects += agent.update_belief(agent.belief_values, -2)
		if agent.belief > 0.75:
			b_i.set_visible(True)
		else:
			b_i.set_visible(False)

		write_objects += [c_i,b_i,text_i]
	return write_objects


def coords(s, ncols):
	return (int(s / ncols), int(s % ncols))


nrows = 30
ncols = 30

# ---------- PART 2:
regionkeys = {'pavement', 'gravel', 'grass', 'sand', 'deterministic'}
regions = dict.fromkeys(regionkeys, {-1})
regions['deterministic'] = range(nrows * ncols)

moveobstacles = []
obstacles = []

agents,tr_ar,building_squares = grid_init(nrows, ncols)
fig.canvas.mpl_connect('key_press_event', on_press)
# ani = FuncAnimation(fig, update_all, frames=10, interval=1250, blit=True, repeat=True)
simulation = Simulation(FuncAnimation(fig, update_all, frames=frames, interval=150, blit=True,repeat=False))
plt.show()
