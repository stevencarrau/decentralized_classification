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
import itertools
# import darpa_sim

# import texfig

class Agent():
    def __init__(self, c_i, label,bad_i, belief=0):
        self.label = label
        self.c_i = c_i
        self.b_i = bad_i
        self.belief = 0  # All agents presumed innocent to begin with

    ## Belief update rule for each agent
    def update_belief(self,belief,bad_idx):
        # TODO: self.belief = max(df[str(i)]['Belief'][agent_idx][)?
        # TODO: print to console or something if belief > 0.8? Or another arbitrary value?
        self.belief = belief[bad_idx][0]


class Simulation():
    def __init__(self, ani):
        self.ani = ani
        self.ani.running = True
        self.state = None  # TODO: maybe use this later?

# ---------- PART 1: Globals
## Fast convergence
with open('AgentPaths_MDP_Fast.json') as json_file:
    data = json.load(json_file)
## Original model -- lots of nominal paths
# with open('AgentPaths_MDP.json') as json_file:
#     data = json.load(json_file)

df = pd.DataFrame(data)
my_dpi = 150
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure(figsize=(3600 / my_dpi, 2000 / my_dpi), dpi=my_dpi)
my_palette = plt.cm.get_cmap("tab10", len(df.index))
categories = [str(d_i) for d_i in df['0'][0]['Id_no']]

belief_y_good = []
frames = len(data)

# image stuff
triggers = ["nominal", "ice_cream_truck", "fire_alarm", "explosion"]
trigger_image_paths = ['pictures/ice_cream.png', 'pictures/fire_alarm.png',
                       'pictures/explosion.png']
trigger_image_xy = (28,2)

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
    elif event.key.lower() == "b":  # trigger explosion
        print("bomb")
    elif event.key.lower() == "i":  # trigger ice cream truck
        print("ice cream")
    elif event.key.lower() == "k":  # trigger fire alarm
        print("fire alarm")
    elif event.key.lower() == "r":  # trigger nominal/normal state
        print("normal state")

    # update simulation animation
    simulation.ani = ani

def update_all(i):
    grid_obj = grid_update(i)
    return grid_obj


def grid_init(nrows, ncols, obs_range):
    global agents
    # fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
    ax = plt.subplot(111)
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
    #names = ["Store A Owner: Andy", "Store B Owner: Barney", "Customer: Chloe", "Customer: Dora", "Customer: Edward",
    #          "Robot"]
    names = ["A", "B", "C", "D", "E", "F"]

    # bad ppl: thanos
    # good ppl: Captain America, Iron man, spiderman, Hulk, Thor
    storeA_squares = [list(range(m,m+5)) for m in range(366,486,30)]
    home_squares = [list(range(m,m+5)) for m in range(380,500,30)]
    storeB_squares = [list(range(m,m+5)) for m in range(823,890,30)]
    building_squares = list(itertools.chain(*home_squares))+list(itertools.chain(*storeA_squares))+list(itertools.chain(*storeB_squares))
    building_doors = [458,472,825]

    for idx, id_no in enumerate(categories):
        # p_t = df[str(0)][id_no]['PublicTargets']
        # color = colors[int(id_no)]
        # color = my_palette(i)
        init_loc = tuple(reversed(coords(df[str(0)][id_no]['AgentLoc'], ncols)))
        # c_i = plt.Circle(init_loc, 0.45, label=names[int(id_no)], color=color)
        c_i = AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no)]), zoom=0.13), xy=init_loc, frameon=False)
        b_i = plt.Circle([init_loc[0]+1,init_loc[1]-1], 0.25, label=names[int(id_no)], color='r')
        # b_i.set_visible(False)
        currAgent = Agent(c_i, names[idx],b_i)
        agents.append(currAgent)

        t_i = None

        # route_x, route_y = zip(*[tuple(reversed(coords(df[str(t)][str(id_no)]['NominalTrace'][s][0],ncols))) for s in df[str(t)][str(id_no)]['NominalTrace']])
        cir_ax = ax.add_artist(currAgent.c_i)
        bad_ax = ax.add_artist(currAgent.b_i)
        ag_array.append([cir_ax,bad_ax])

        if int(id_no) < len(trigger_image_paths):
            trigger_image_path_index = int(id_no) % len(trigger_image_paths)
            t_i = AnnotationBbox(OffsetImage(plt.imread(trigger_image_paths[trigger_image_path_index]), zoom=0.08),
                             xy=trigger_image_xy, frameon=False)
            t_i.set_label(trigger_image_paths[trigger_image_path_index])
            trigger_ax = ax.add_artist(t_i)
            tr_array.append([trigger_ax])
        # legend = plt.legend(handles=cir_ax, loc=4, fontsize='small', fancybox=True)


        # fill in buildings
        # # store A
        # ax.fill([5 + 0.5, 10 + 0.5, 10 + 0.5, 5 + 0.5],
        #         [12 - 0.5, 12 - 0.5, 15 + 0.5, 15 + 0.5], color=brown, alpha=0.9)
        #
        # # Home
        # ax.fill([18 + 0.5, 23 + 0.5, 23 + 0.5, 18 + 0.5],
        #         [12 - 0.5, 12 - 0.5, 15 + 0.5, 15 + 0.5], color=brown, alpha=0.9)

        # # store B
        # ax.fill([12 + 0.5, 17 + 0.5, 17 + 0.5, 12 + 0.5],
        #         [27 - 0.5, 27 - 0.5, 29 + 0.5, 29 + 0.5], color=brown, alpha=0.9)
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

        # for k in p_t:
        # 	s_c = coords(k, ncols)
        # 	ax.fill([s_c[1]+0.4, s_c[1]-0.4, s_c[1]-0.4, s_c[1]+0.4], [s_c[0]-0.4, s_c[0]-0.4, s_c[0]+0.4, s_c[0]+0.4], color=color, alpha=0.9)
        i += 1

    # legend = plt.legend(handles=ag_array, loc=4, fontsize='small', fancybox=True)
    # ax.legend()
    return ag_array,tr_array,building_squares


def grid_update(i):
    global ax_ar,tr_ar, df, ncols, obs_range,building_squares, agents, simulation
    write_objects = []

    if not simulation.ani.running:
        return write_objects

    active_event = df[str(i)]['Event']
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


    for agent_idx, agent in enumerate(agents):
        # c_i, l_i, p_i,p_2 = a_x
        c_i = agent.c_i
        b_i = agent.b_i
        loc = tuple(reversed(coords(df[str(i)][str(agent_idx)]['AgentLoc']-30, ncols)))
        # Use below line if you're working with circles
        b_i.set_center([loc[0]+1,loc[1]-1])

        # Use this line if you're working with images
        c_i.xy = loc
        c_i.xyann = loc
        c_i.xybox = loc

        if df[str(i)][str(agent_idx)]['AgentLoc'] in building_squares:
            c_i.offsetbox.image.set_alpha(0.35)
        else:
            c_i.offsetbox.image.set_alpha(1.0)

        ## TODO -- Insert belief checking module here -- loop through each agent then check belief across possible models
        agent.c_i = c_i
        agent.update_belief(df[str(i)]['Belief'][agent_idx],-1)
        if agent.belief > 0.75:
            print('Bad')
            b_i.set_visible(True)
        else:
            b_i.set_visible(False)

        write_objects += [c_i,b_i]
    return write_objects


def coords(s, ncols):
    return (int(s / ncols), int(s % ncols))


nrows = 30
ncols = 30

# ---------- PART 2:
regionkeys = {'pavement', 'gravel', 'grass', 'sand', 'deterministic'}
regions = dict.fromkeys(regionkeys, {-1})
regions['deterministic'] = range(nrows * ncols)
# gwg = Gridworld(initial=[0],nrows=nrows,ncols=ncols,regions=regions)
# gwg.render()
# gwg.draw_state_labels()

moveobstacles = []
obstacles = []

# #4 agents larger range
obs_range = 6

# con_dict = con_ar = con_init()
# bel_lines = belief_chart_init()
ax_ar,tr_ar,building_squares = grid_init(nrows, ncols, obs_range)

# ani = FuncAnimation(fig, update_all, frames=10, interval=1000, blit=True, repeat=False)
# plt.show()
# ani.save('6_agents_pink_bad.mp4', writer=writer)
fig.canvas.mpl_connect('key_press_event', on_press)
# ani = FuncAnimation(fig, update_all, frames=10, interval=1250, blit=True, repeat=True)
simulation = Simulation(FuncAnimation(fig, update_all, frames=frames, interval=150, blit=True,repeat=False))
plt.show()
