import json
import time

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from math import ceil,floor
from math import pi
import math
from ast import literal_eval
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)
import numpy as np
from gridworld import *
import itertools
from darpa_model import ERSA_Env,track_outs
import json
import argparse

class Agent():
	def __init__(self, c_i, label, char_name, bad_i,mdp, state, t_i, agent_idx=0):
		self.label = label
		self.char_name = char_name
		self.c_i = c_i
		self.b_i = bad_i
		self.t_i = t_i
		self.belief_values = np.ones((len(Simulation.init_states),1))/len(Simulation.init_states)
		self.belief = 0  # All agents presumed innocent to begin with
		self.agent_idx = agent_idx
		self.state = state
		self.mdp = mdp
		self.track_queue = []

	def likelihood(self,a,next_s,mc_dict):
		return np.array([m_i[(self.state, next_s)] for m_i in mc_dict[a]]).reshape((-1, 1))

	def update_value(self,a,next_s):
		self.belief_values = belief_update(self.belief_values,self.likelihood(a,next_s,Simulation.mc_dict))

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
			f = lambda i: belief[i]
			if max(belief)> 0.75:
				if max(range(len(belief)), key=f) == self.agent_idx:
					self.belief_text.set_color('green')
				else:
					self.belief_text.set_color('red')
				if self.belief > 0.75:
					self.belief_line.set_color('red')
					self.belief_fill.set_color('red')
				else:
					self.belief_line.set_color('green')
					self.belief_fill.set_color('green')
			else:
				self.belief_line.set_color('yellow')
				self.belief_fill.set_color('yellow')
				self.belief_text.set_color('black')
			return [self.belief_line, self.belief_fill, self.belief_artist,self.belief_text]
		return None

	def init_belief_plt(self, l_i, l_f, l_a,l_t):
		self.belief_line = l_i
		self.belief_line.set_visible(False)
		self.belief_fill = l_f
		self.belief_fill.set_visible(False)
		self.belief_artist = l_a
		self.belief_artist.set_visible(False)
		self.belief_text = l_t
		self.belief_artist.set_visible(False)

	def activate_belief_plt(self):
		self.belief_line.axes.set_visible(True)
		self.belief_line.set_visible(True)
		self.belief_fill.set_visible(True)
		self.belief_artist.set_visible(True)

	def deactivate_belief_plt(self):
		self.belief_line.axes.set_visible(False)
		self.belief_line.set_visible(False)
		self.belief_fill.set_visible(False)
		self.belief_artist.set_visible(False)



class Simulation():
	init_states = [0,1,2,3,4,5]
	mc_dict = None

	def on_press(event):
		ani = Singleton.instance.ani

		if event.key.isspace():
			if ani.running:
				ani.event_source.stop()
			else:
				ani.event_source.start()
			ani.running ^= True

		elif event.key.lower() == "j":  # trigger nominal
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
		elif event.key.lower() == "+":  # add agent
			Singleton.instance.add_agent()
		elif event.key.lower() == "-":  # remove agent
			Singleton.instance.remove_agent()
		elif event.key.lower() == "m":
			ani.moving ^= True

		# update simulation animation
		Singleton.instance.ani = ani

	def on_click(event):
		ani = Singleton.instance.ani
		if event.button == 1:
			ani.observer_loc = tuple(reversed((floor(event.xdata), floor(event.ydata))))
		elif event.button == 3:
			ani.remove_loc = tuple(reversed((floor(event.xdata), floor(event.ydata))))
		# print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))

		# update simulation animation
		Singleton.instance.ani = ani

	def __init__(self, ani, gwg, ax, agents, tr_ar, observable_regions, building_squares, nrows, ncols):
		self.ani = ani
		self.ani.running = True
		self.ani.moving = False
		self.ani.event = 0
		self.ani.observer_loc = None
		self.ani.remove_loc = None
		self.state = None
		self.ax = ax
		self.observable_regions = observable_regions
		self.observable_states = set(gwg.states) - set(gwg.obstacles)
		self.observable_set = set(gwg.states) - set(gwg.obstacles)
		self.observable_artists = []
		for h_s in self.observable_set:
			h_loc = tuple(reversed(coords(h_s, ncols)))
			self.observable_artists.append(ax.fill([h_loc[0] - 0.5, h_loc[0] + 0.5, h_loc[0] + 0.5, h_loc[0] - 0.5],
												   [h_loc[1] - 0.5, h_loc[1] - 0.5, h_loc[1] + 0.5, h_loc[1] + 0.5],
												   color='gray', alpha=0.00)[0])

		self.observers = []
		self.observers_artists = []
		self.agents = agents
		self.tr_ar = tr_ar
		self.building_squares = building_squares
		self.nrows = nrows
		self.ncols = ncols
		self.active_agents = 0    # initialize simulation to not have any agents active
		self.num_agents = len(self.agents)
		self.categories = range(self.num_agents)


	def blit_viewable_states(self):
		write_objects = []
		for o_s, o_a in zip(self.observable_set, self.observable_artists):
			if o_s in self.observable_states:
				o_a.set_alpha(0.00)
				write_objects += [o_a]
			else:
				o_a.set_alpha(0.25)
				write_objects += [o_a]
		for a_i in self.observers_artists:
			a_i.set_zorder(10)
			write_objects += [a_i]
		return write_objects

	def add_observer(self, obs_state):
		self.observers.append(obs_state)
		new_observables = set()
		for o_i in self.observers:
			[new_observables.add(s) for s in self.observable_regions[str(o_i)]]
		self.observable_states = new_observables
		write_objects = self.blit_viewable_states()
		o_loc = tuple(reversed(coords(obs_state, self.ncols)))
		o_x = self.ax.fill([o_loc[0] - 0.5, o_loc[0] + 0.5, o_loc[0] + 0.5, o_loc[0] - 0.5],
					 [o_loc[1] - 0.5, o_loc[1] - 0.5, o_loc[1] + 0.5, o_loc[1] + 0.5],
					 color='green', alpha=0.50)[0]
		self.observers_artists.append(o_x)
		write_objects += [o_x]
		return write_objects


	def remove_observer(self, obs_state):
		if obs_state in self.observers:
			o_x = self.observers_artists.pop(self.observers.index(obs_state))
			o_x.remove()
			self.observers.remove(obs_state)
		new_observables = set()
		for o_i in self.observers:
			[new_observables.add(s) for s in self.observable_regions[str(o_i)]]
		self.observable_states = new_observables
		write_objects = self.blit_viewable_states()
		return write_objects

	def add_agent(self):
		if 0 <= self.active_agents < self.num_agents:
			self.active_agents += 1

	def remove_agent(self):
		if 0 < self.active_agents <= self.num_agents:
			self.active_agents -= 1


class Singleton():
	instance = None

	def __init__(self, ani, gwg, ax, agents, tr_ar, observable_regions, building_squares, nrows, ncols):
		# make sure there's only one instance
		if Singleton.instance is None:
			Singleton.instance = Simulation(ani, gwg, ax, agents, tr_ar, observable_regions, building_squares, nrows, ncols)
		else:
			print("Already have one instance of the simulation running!")



def update_all(i):
	grid_obj = grid_update(i)
	return grid_obj


def belief_update(belief,likelihood):
	new_belief = np.multiply(belief,likelihood)
	new_belief = new_belief/np.sum(new_belief)
	return new_belief


def grid_init(nrows, ncols, desiredIndices):
	# bad ppl: thanos (threat MDP)
	# good ppl: Captain A (Store A MDP), Iron man (home MDP), black widow (store B MDP),
	# Hulk (repairman MDP), Thor (shopper MDP)
	agent_types = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F'}

	init_states = [ind[1] for ind in desiredIndices]
	agents = []   # use "None"s as placeholders
	agent_image_paths = ['pictures/captain_america.png', 'pictures/black_widow.png', 'pictures/hulk.png',
						 'pictures/thor.png', 'pictures/thanos.png', 'pictures/ironman.png']
	agent_character_names = ['Captain America', 'Black Widow', 'Hulk', 'Thor', 'Thanos', 'Ironman']
	names = ["Store A Owner", "Store B Owner", "Repairman", "Shopper", "Suspicious", "Home Owner"]

	mdp_list = ERSA_Env()

	# image stuff
	triggers = ["nominal", "ice_cream_truck", "fire_alarm", "explosion"]
	trigger_image_paths = 3 * ['pictures/ice_cream.png'] + 2 * ['pictures/fire_alarm.png']
	trigger_image_xy = [(8, 19), (15, 19), (22, 19), (4.5, 12), (13, 27)]

	categories = range(len(init_states))  # [str(d_i) for d_i in df['0'][0]['Id_no']]
	# fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
	ax = plt.subplot2grid((len(categories), 7), (0, 3), rowspan=len(categories), colspan=4)
	t = 0

	# legend stuff
	# legend_text = "A: {:<20} Store A Owner \t B: {:<20}Store B Owner\tC: {:<20}Repairman \n D: {:<20}Shopper\t E: {:<20}Suscipious \t F: {:<20}Home Owner".format()
	legend_text = "A: {:<25} B: {:<25} C: {} \nD: {:<25} E: {:<25} F: {}".format('Store A Owner','Store B Owner','Repairman','Shopper','Suscipious','Home Owner')
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

	storeA_squares = [list(range(m,m+5)) for m in range(366,486,30)]
	home_squares = [list(range(m,m+5)) for m in range(380,500,30)]
	storeB_squares = [list(range(m,m+5)) for m in range(823,890,30)]
	building_squares = list(itertools.chain(*home_squares))+list(itertools.chain(*storeA_squares))+list(itertools.chain(*storeB_squares))
	building_doors = [458,472,825]

	# set up agents
	for idx, id_no in enumerate(desiredIndices):
		# p_t = df[str(0)][id_no]['PublicTargets']
		# color = colors[int(id_no)]
		# color = my_palette(i)
		samp_out = mdp_list[idx].sample(id_no[1], 0)
		track_init = track_outs((id_no[1], samp_out))
		init_loc = tuple(reversed(coords(track_init[0] - 30, ncols)))
		# c_i = plt.Circle(init_loc, 0.45, label=names[int(id_no)], color=color)
		t_i = plt.text(x=init_loc[0], y=init_loc[1], s=names[int(id_no[0])], fontsize='xx-small')
		t_i.set_visible(False)  # don't show the labels until the agent is added
		c_i = AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no[0])]), zoom=0.13),
							 xy=init_loc, frameon=False)
		c_i.set_label(names[idx])
		c_i.set_visible(False)
		b_i = plt.Circle([init_loc[0] + 1, init_loc[1] - 1], 0.25, label=names[int(id_no[0])], color='r')
		b_i.set_visible(False)
		currAgent = Agent(c_i=c_i, label=names[int(id_no[0])], char_name=id_no[0], \
						  bad_i=b_i, mdp=mdp_list[int(id_no[0])], state=id_no[1], t_i=t_i,agent_idx=id_no[0])
		agents.append(currAgent)
		# currAgentIndex = desiredIndices[idx][1]
		# agents.insert(currAgentIndex, currAgent)
		# agents.pop(currAgentIndex-1)   # remove each "None" as we insert a new agent

	for idx, id_no in enumerate(agents):
		currAgent = agents[idx]
		# route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['NominalTrace'][s][0],ncols))) for s in df[str(t)][str(id_no)]['NominalTrace']])
		cir_ax = ax.add_artist(currAgent.c_i)
		bad_ax = ax.add_artist(currAgent.b_i)
		ag_array.append([cir_ax, bad_ax])

	for t_p,t_l in zip(trigger_image_paths,trigger_image_xy):
		t_i = AnnotationBbox(OffsetImage(plt.imread(t_p), zoom=0.04),
						 xy=t_l, frameon=False)
		t_i.set_visible(False)
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

	# Electric Utility Control Box
	ax.fill([12 + 0.5, 14 + 0.5, 14 + 0.5, 12 + 0.5],
			[5 + 0.5, 5 + 0.5, 7 + 0.5, 7 + 0.5], color=blue, alpha=0.9)

	# Street
	ax.fill([-1.0 + 0.5, 29.5 + 0.5, 29.5 + 0.5, -1.0 + 0.5],
			[18 + 0.5, 18 + 0.5, 24 + 0.5, 24 + 0.5], color=gray, alpha=0.9)


	# make the legend
	legend_dict = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.00, 1.08, legend_text, transform=ax.transAxes, fontsize=8,
			verticalalignment='top', bbox=legend_dict)


	## Plot for belief charts
	ax_list = []
	angles = [n / float(len(agent_types)) * 2 * pi for n in range(len(agent_types))]
	angles += angles[:1]
	for idx, id_no in enumerate(desiredIndices):
		belief = np.zeros((len(agent_types),)).tolist()
		belief[id_no[0]] = 1
		ax_list.append(plt.subplot2grid((len(desiredIndices), 7), (2*int(idx/2), idx % 2+1*(idx%2)), rowspan=2, colspan=1, polar=True))
		ax_list[-1].set_theta_offset(pi / 2)
		ax_list[-1].set_theta_direction(-1)
		ax_list[-1].set_ylim(0, 100)
		if idx ==0:
			# plt.xticks(angles[:-1],['A','B','C','D','E','F'], color='grey', size=8)
			plt.xticks(angles[:-1], "", color='grey', size=8)
		else:
			plt.xticks(angles[:-1], "", color='grey', size=8)
		# for col, xtick in enumerate(ax_list[-1].get_xticklabels()):
		# 	xtick.set(color=my_palette(col), fontweight='bold', fontsize=16)
		ax_list[-1].set_rlabel_position(0)
		ax_list[-1].tick_params(pad=-7.0)
		ax_list[-1].set_rlabel_position(45)
		plt.yticks([50, 100], ["", ""], color="grey", size=7)
		l, = ax_list[-1].plot([], [], color='green', linewidth=2, linestyle='solid')
		l_f, = ax_list[-1].fill([], [], color='green', alpha=0.4)
		val = [75 * b_i + 25 for b_i in belief]
		val += val[:1]
		ax_list[-1].fill(angles,val, color='grey', alpha=0.4)
		ax_list[-1].spines["bottom"] = ax_list[-1].spines["inner"]
		# l.axes.set_visible(False)
		agent_pic = AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no[0])]), zoom=0.20), xy=(0, 0),frameon=False)
		agent_pic.xyann = (5.0,175)
		agent_pic.xybox = (5.0,175)
		agent_txt = plt.text(x=4.25,y=215,s=agent_types[id_no[0]], fontsize=18)
		l_a = ax_list[-1].add_artist(agent_pic)
		agents[idx].init_belief_plt(l, l_f, l_a,agent_txt)


	return agents,tr_array,building_squares,ax


def grid_update(i):
	# global df, obs_range
	simulation = Singleton.instance
	tr_ar = simulation.tr_ar
	agents = simulation.agents[:simulation.active_agents]
	leftover_agents = simulation.agents[simulation.active_agents:]
	ncols = simulation.ncols
	building_squares = simulation.building_squares
	write_objects = []


	if simulation.ani.observer_loc is not None or simulation.ani.remove_loc is not None:
		if simulation.ani.observer_loc is not None:
			write_objects += simulation.add_observer(coord2state(simulation.ani.observer_loc,ncols))
			simulation.ani.observer_loc = None
		if simulation.ani.remove_loc is not None:
			write_objects += simulation.remove_observer(coord2state(simulation.ani.remove_loc, ncols))
			simulation.ani.remove_loc = None
	else:
		write_objects += simulation.blit_viewable_states()

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
		if len(agent.track_queue) == 0:
			next_s = agent.mdp.sample(agent.state,simulation.ani.event)
			agent.track_queue += track_outs((agent.state,next_s))
			time.sleep(0.05)
			if agent.track_queue[0] in simulation.observable_states:
				agent.update_value(simulation.ani.event,next_s)
				write_objects += agent.update_belief(agent.belief_values, -2)
			agent.state = next_s
		c_i = agent.c_i
		c_i.set_visible(True)
		b_i = agent.b_i
		agent.activate_belief_plt()
		b_i.set_visible(True)
		text_i = agent.t_i
		if simulation.ani.moving:
			agent_pos = agent.track_queue.pop(0)
		else:
			agent_pos = agent.track_queue[0]
		# text_i = agent.t_i
		# agent_pos = agent.track_queue.pop(0)
		loc = tuple(reversed(coords(agent_pos-30, ncols)))
		# Use below line if you're working with circles
		b_i.set_center([loc[0]+1,loc[1]-1])

		# Use this line if you're working with images
		c_i.xy = loc
		c_i.xyann = loc
		c_i.xybox = loc

		# update text positions
		# text_i.set_visible(False)   # remove old label
		# text_i.set_position((loc[0]-1.5, loc[1]+1.5))  # move label to new (x, y)
		# text_i.set_visible(True)
		# agent.t_i = text_i

		if agent_pos in building_squares:
			c_i.offsetbox.image.set_alpha(0.35)
		else:
			c_i.offsetbox.image.set_alpha(1.0)

		agent.c_i = c_i
		if agent.belief > 0.75:
			b_i.set_visible(True)
		else:
			b_i.set_visible(False)
		write_objects += [c_i,b_i]

	for agent_idx, agent in enumerate(leftover_agents):
		c_i = agent.c_i
		c_i.set_visible(False)
		b_i = agent.b_i
		b_i.set_visible(False)
		text_i = agent.t_i
		text_i.set_visible(False)
		agent.deactivate_belief_plt()
		agent.c_i = c_i
		agent.b_i = b_i
		# agent.t_i = text_i
		write_objects += [c_i,b_i]



	# Update everything TODO: is this necessary?
	Singleton.instance = simulation
	Singleton.instance.tr_ar = tr_ar

	return write_objects


def coords(s, ncols):
	return (int(s / ncols), int(s % ncols))


def coord2state(coords,ncols):
	return int(coords[0]*ncols)+int(coords[1])

def get_agent_indices(args):
	names = ["Store A Owner", "Store B Owner", "Repairman", "Shopper", "Suspicious", "Home Owner"]
	numAgents = len(args)-1
	inputs = []

	# form of each input: ({agent type},{desired index in agents list})
	for agentIdx, arg in enumerate(args):
		if agentIdx == 0:
			continue

		currInput = []
		for num in arg.split(","):
			currNum = ""
			for char in num:
				if char != ")" and char != "(":
					currNum += char
			currInput.append(int(currNum))
		inputs.append(tuple(currInput))


	# verify arguments
	reqIndices = [i for i in range(numAgents)]

	# for input in inputs:
	# 	agentIdx = input[0]
	# 	if 0 <= agentIdx < numAgents and agentIdx in range(numAgents):
	# 		reqIndices.remove(agentIdx)
	# 	else:
	# 		print("Invalid arguments")
	# 		sys.exit(1)
	#
	# if len(reqIndices) != 0:
	# 	print("Invalid arguments")
	# 	sys.exit(1)

	return sorted(inputs)

def main():
	# ---------- PART 1:
	# grab arguments of format: 'Store A Owner','Store B Owner','Repairman','Shopper','Suscipious','Home Owner'
	agent_indices = get_agent_indices(sys.argv)
	event_names = {0: 'nominal', 1: 'iceA', 2: 'iceB', 3: 'iceC', 4: 'alarmA', 5: 'alarmB', 6: 'alarmG'}

	mdp_list = ERSA_Env()
	mc_dict = dict()
	env_states = [0, 1, 2, 3, 4, 5, 6, 7]
	for a in event_names.keys():
		mc_dict.update({a:[m.construct_MC(dict([[s,[a]] for s in env_states])) for m in mdp_list]})

	mc_dict = dict()
	for a in event_names.keys():
		mc_dict.update({a:[m.construct_MC(dict([[s,[a]] for s in env_states])) for m in mdp_list]})

	Simulation.mc_dict = mc_dict

	with open('VisibleStates.json') as json_file:
		observable_regions = json.load(json_file)


	my_dpi = 150
	Writer = matplotlib.animation.writers['ffmpeg']
	writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
	fig = plt.figure(figsize=(3600 / my_dpi, 2000 / my_dpi), dpi=my_dpi)

	belief_y_good = []
	frames = 500

	# ---------- PART 2:
	nrows = 30
	ncols = 30
	regionkeys = {'pavement', 'gravel', 'grass', 'sand', 'deterministic'}
	regions = dict.fromkeys(regionkeys, {-1})
	regions['deterministic'] = range(nrows * ncols)

	moveobstacles = []
	obstacles = []

	agents, tr_ar, building_squares, ax = grid_init(nrows, ncols, agent_indices)
	gwg = Gridworld([0], nrows=nrows, ncols=ncols,regions=regions,obstacles=building_squares)
	fig.canvas.mpl_connect('key_press_event', Simulation.on_press)
	fig.canvas.mpl_connect('button_press_event', Simulation.on_click)
	# ani = FuncAnimation(fig, update_all, frames=10, interval=1250, blit=True, repeat=True)
	anim = FuncAnimation(fig, update_all, frames=frames, interval=10, blit=False,repeat=False)
	Singleton(anim, gwg, ax, agents, tr_ar, observable_regions, building_squares, nrows, ncols)
	plt.show()


if __name__ == '__main__':
	main()