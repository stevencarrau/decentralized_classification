import itertools
import json
import sys
from enum import Enum
from math import floor
from math import pi
from collections import deque

import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

from darpa_model import ERSA_Env, track_outs
from gridworld import *
from agent import Agent
from sensor import Sensor
from util import Util

matplotlib.use('Qt5Agg')

frames = 500
np.random.seed(0)


class SensorState(Enum):
    REMOVE_SENSOR = 1
    ADD_SENSOR = 2
    DO_NOTHING = 3




class Simulation:
    init_states = [0, 1, 2, 3, 4, 5]
    mc_dict = None

    def __init__(self, ani, gwg, ax, agents, tr_ar, rl_ar, observable_regions, building_squares, nrows, ncols,
                 counter_text):
        self.ani = ani
        self.ani.running = True
        self.ani.moving = False
        self.ani.event = 0
        self.ani.sensor_state = SensorState.DO_NOTHING
        self.state = None
        self.ax = ax
        self.observable_regions = observable_regions
        self.observable_states = set(gwg.states) - set(gwg.obstacles)
        self.observable_set = set(gwg.states) - set(gwg.obstacles)
        self.observable_artists = []
        for h_s in self.observable_set:
            h_loc = tuple(reversed(Util.coords(h_s, ncols)))
            self.observable_artists.append(ax.fill([h_loc[0] - 0.5, h_loc[0] + 0.5, h_loc[0] + 0.5, h_loc[0] - 0.5],
                                                   [h_loc[1] - 0.5, h_loc[1] - 0.5, h_loc[1] + 0.5, h_loc[1] + 0.5],
                                                   color='gray', alpha=0.00)[0])

        self.observers_artists = []
        self.sensors = []
        self.agents = agents
        self.tr_ar = tr_ar
        self.rl_ar = rl_ar
        self.building_squares = building_squares
        self.nrows = nrows
        self.ncols = ncols
        self.num_agents = len(self.agents)
        self.active_agents = self.num_agents  # 0    # initialize simulation to not have any agents active
        self.categories = range(self.num_agents)
        self.time_step = -1
        self.counter_text = counter_text
        self.pause_counter = 0
        self.pause_flag = False
        self.gwg = gwg

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

    def add_agent(self):
        if 0 <= self.active_agents < self.num_agents:
            self.active_agents += 1

    def remove_agent(self):
        if 0 < self.active_agents <= self.num_agents:
            self.active_agents -= 1

    def update_sensor_locations(self):
        write_objects=[]
        if len(self.sensors) != 0:
            obs_states = set()
            for s_i in self.sensors:
                write_objects += [s_i.update_sensor()]
                obs_states.update(s_i.observable_states)
            self.observable_states = obs_states
        if self.sensors is not None and self.ani.sensor_state != SensorState.DO_NOTHING:
            if len(self.sensors) !=0 and self.ani.sensor_state == SensorState.ADD_SENSOR:
                write_objects += [self.sensors[-1].sensor_artist]
            # else:
            #     write_objects += Sensor.remove_observer(curr_sensor_loc)
            self.ani.sensor_state = SensorState.DO_NOTHING

        write_objects += self.blit_viewable_states()
        return write_objects


class SimulationRunner:
    instance = None

    def __init__(self, ani, gwg, ax, agents, tr_ar, rl_ar, observable_regions, building_squares, nrows, ncols,
                 counter_text):
        # make sure there's only one instance
        if SimulationRunner.instance is None:
            SimulationRunner.instance = Simulation(ani, gwg, ax, agents, tr_ar, rl_ar, observable_regions,
                                                   building_squares,
                                                   nrows, ncols, counter_text)
        else:
            print("Already have one instance of the simulation running!")

    @staticmethod
    def update_all(i):
        grid_obj = SimulationRunner.grid_update(i)
        return grid_obj

    @staticmethod
    def grid_init(nrows, ncols, desiredIndices, agent_track_queues=None):
        # bad ppl: thanos (threat MDP)
        # good ppl: Captain A (Store A MDP), Iron man (home MDP), black widow (store B MDP),
        # Hulk (repairman MDP), Thor (shopper MDP)
        agent_types = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

        init_states = [ind[1] for ind in desiredIndices]
        agents = []  # use "None"s as placeholders
        agent_image_paths = ['pictures/captain_america', 'pictures/black_widow', 'pictures/hulk',
                             'pictures/thor', 'pictures/thanos', 'pictures/ironman']
        path_augments = ['.png','-uncertain.png','-random.png']
        random_image_paths = ['pictures/rnd_level1.png', 'pictures/rnd_level2.png', 'pictures/rnd_level3.png']
        agent_character_names = ['Captain America', 'Black Widow', 'Hulk', 'Thor', 'Thanos', 'Ironman']
        names = ["Store A Owner", "Store B Owner", "Repairman", "Shopper", "Suspicious", "Home Owner"]

        mdp_list, state_list, state_keys = ERSA_Env()

        # image stuff
        triggers = ["nominal", "ice_cream_truck", "fire_alarm", "explosion"]
        trigger_image_paths = 3 * ['pictures/ice_cream.png'] + 2 * ['pictures/fire_alarm.png']
        trigger_image_xy = [(8, 19), (15, 19), (22, 19), (4.5, 12), (13, 27)]

        categories = range(len(init_states))  # [str(d_i) for d_i in df['0'][0]['Id_no']]
        # fig_new = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
        ax = plt.subplot2grid((len(categories), 7), (0, 3), rowspan=len(categories), colspan=4)
        t = 0

        # legend stuff
        legend_text = "A: {:<25} B: {:<25} C: {} \nD: {:<25} E: {:<25} F: {}".format('Store A Owner', 'Store B Owner',
                                                                                     'Repairman', 'Shopper',
                                                                                     'Suscipious',
                                                                                     'Home Owner')
        plt.xticks(range(ncols), '')
        plt.yticks(range(nrows), '')
        ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(-0.5, nrows - 0.5)
        ax.invert_yaxis()
        ag_array = []
        tr_array = []
        rl_array = []
        rl_indicator_loc = (28, 2.5)  # Indicator of random location
        plt.grid(which="minor", ls="-", lw=1)

        # define a few colors
        brown = (255.0 / 255.0, 179.0 / 255.0, 102.0 / 255.0)
        gray = (211.0 / 255.0, 211.0 / 255.0, 211.0 / 255.0)
        blue = (173.0 / 255.0, 216.0 / 255.0, 230.0 / 255.0)
        mustard = (255.0 / 255.0, 225.0 / 255.0, 77.0 / 255.0)

        storeA_squares = [list(range(m, m + 5)) for m in range(366, 486, 30)]
        home_squares = [list(range(m, m + 5)) for m in range(380, 500, 30)]
        storeB_squares = [list(range(m, m + 5)) for m in range(823, 890, 30)]
        building_squares = list(itertools.chain(*home_squares)) + list(itertools.chain(*storeA_squares)) + list(
            itertools.chain(*storeB_squares))
        building_doors = [458, 472, 825]

        # set up agents
        for idx, id_no in enumerate(desiredIndices):
            samp_out = Util.prod2state(mdp_list[idx].sample(Util.state2prod(id_no[1], 0, state_keys), 0), state_list)
            track_init = track_outs((id_no[1], samp_out))
            init_loc = tuple(reversed(Util.coords(track_init[0] - 30, ncols)))
            # c_i = plt.Circle(init_loc, 0.45, label=names[int(id_no)], color=color)
            t_i = plt.text(x=init_loc[0], y=init_loc[1], s=names[int(id_no[0])], fontsize='xx-small')
            t_i.set_visible(False)  # don't show the labels until the agent is added
            c_set = []
            for r_j in range(3):
                c_i = AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no[0])]+path_augments[r_j]), zoom=0.13),
                                 xy=init_loc, frameon=False)
                c_i.set_label(names[idx])
                c_i.set_visible(False)
                c_set.append(c_i)
            b_i = plt.Circle([init_loc[0] + 1, init_loc[1] - 1], 0.25, label=names[int(id_no[0])], color='r')
            b_i.set_visible(False)
            currAgent = Agent(c_set=c_set, label=names[int(id_no[0])], char_name=id_no[0],
                              bad_i=b_i, mdp=mdp_list[int(id_no[0])], state=Util.state2prod(id_no[1], 0, state_keys),
                              t_i=t_i,
                              agent_idx=id_no[0], states=state_list,mc_dict=Simulation.mc_dict,state_keys=state_keys)
            # set pre-loaded tracks if desired
            if agent_track_queues is not None:
                currAgent.track_queue = agent_track_queues[idx]
            agents.append(currAgent)

        for idx, id_no in enumerate(agents):
            currAgent = agents[idx]
            ag_elem = []
            for c_j in currAgent.c_set:
                ag_elem += [ax.add_artist(c_j)]
            ag_elem += [ax.add_artist(currAgent.b_i)]
            ag_array.append(ag_elem)

        for t_p, t_l in zip(trigger_image_paths, trigger_image_xy):
            t_i = AnnotationBbox(OffsetImage(plt.imread(t_p), zoom=0.04),
                                 xy=t_l, frameon=False)
            t_i.set_visible(False)
            trigger_ax = ax.add_artist(t_i)
            tr_array.append([trigger_ax])

        for r_p in random_image_paths:
            r_i = AnnotationBbox(OffsetImage(plt.imread(r_p), zoom=0.1),
                                 xy=rl_indicator_loc, frameon=False, annotation_clip=False)
            r_i.set_visible(False)
            rand_ax = ax.add_artist(r_i)
            rl_array.append([rand_ax])

        for h_s in building_squares:
            h_loc = tuple(reversed(Util.coords(h_s, ncols)))
            ax.fill([h_loc[0] - 0.5, h_loc[0] + 0.5, h_loc[0] + 0.5, h_loc[0] - 0.5],
                    [h_loc[1] - 0.5, h_loc[1] - 0.5, h_loc[1] + 0.5, h_loc[1] + 0.5], color=brown, alpha=0.8)

        for b_d in building_doors:
            b_loc = tuple(reversed(Util.coords(b_d, ncols)))
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

        counter_text = plt.text(1.00, -0.08, '{}'.format(0), transform=ax.transAxes, fontsize=18)
        ## Plot for belief charts
        ax_list = []
        angles = [n / float(len(agent_types)) * 2 * pi for n in range(len(agent_types))]
        angles += angles[:1]
        for idx, id_no in enumerate(desiredIndices):
            belief = np.zeros((len(agent_types),)).tolist()
            belief[id_no[0]] = 1
            ax_list.append(
                plt.subplot2grid((len(desiredIndices), 7), (2 * int(idx / 2), idx % 2 + 1 * (idx % 2)), rowspan=2,
                                 colspan=1, polar=True))
            ax_list[-1].set_theta_offset(pi / 2)
            ax_list[-1].set_theta_direction(-1)
            ax_list[-1].set_ylim(0, 100)
            if idx == 0:
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
            ax_list[-1].fill(angles, val, color='grey', alpha=0.4)
            ax_list[-1].spines["bottom"] = ax_list[-1].spines["inner"]
            # l.axes.set_visible(False)
            l_set = []
            for c_j in range(3):
                agent_pic = AnnotationBbox(OffsetImage(plt.imread(agent_image_paths[int(id_no[0])]+path_augments[c_j]), zoom=0.20), xy=(0, 0),
                                       frameon=False)
                agent_pic.xyann = (5.0, 175)
                agent_pic.xybox = (5.0, 175)
                l_set += [ax_list[-1].add_artist(agent_pic)]
            agent_txt = plt.text(x=4.25, y=215, s=agent_types[id_no[0]], fontsize=18)
            agents[idx].init_belief_plt(l, l_f, l_set, agent_txt)

        return agents, tr_array, rl_array, building_squares, ax, counter_text

    @staticmethod
    def grid_update(i):
        global frames
        if i == frames - 1:
            plt.close()
        # plt.savefig('video_data/{:04d}.png'.format(i), bbox_inches='tight')
        simulation = SimulationRunner.instance
        tr_ar = simulation.tr_ar
        rl_ar = simulation.rl_ar
        agents = simulation.agents[:simulation.active_agents]
        leftover_agents = simulation.agents[simulation.active_agents:]
        ncols = simulation.ncols
        building_squares = simulation.building_squares

        write_objects = simulation.update_sensor_locations()

        # If not running don't update agents
        if not simulation.ani.running:
            simulation.counter_text.set_text('{}'.format(simulation.time_step - 1))
            write_objects += [simulation.counter_text]
            return write_objects

        active_event = simulation.ani.event
        # if simulation.time_step > 5 and simulation.time_step - 1 < 15:
        #     active_event = 2
        # elif simulation.time_step > 14 and simulation.time_step - 1 < 21:
        #     active_event = 6
        # elif simulation.time_step > 20 and simulation.time_step - 1 < 31:
        #     active_event = 4
        # elif simulation.time_step > 30 and simulation.time_step - 1 < 36:
        #     active_event = 1
        # elif simulation.time_step > 35 and simulation.time_step - 1 < 41:
        #     active_event = 2
        #     for a_i in agents:
        #         a_i.alpha=1
        # elif simulation.time_step > 41 and simulation.time_step - 1 < 48:
        #     active_event = 4
        # else:
        #     active_event = 0
        if active_event == 0:
            for t_i in tr_ar:
                t_i[0].set_visible(False)
                write_objects += t_i
        else:
            for ind, t_i in enumerate(tr_ar):
                if active_event == ind + 1:
                    t_i[0].set_visible(True)
                    write_objects += t_i
                else:
                    t_i[0].set_visible(False)
                    write_objects += t_i
            if active_event == 6:
                tr_ar[-1][0].set_visible(True)
                tr_ar[-2][0].set_visible(True)
                write_objects += [tr_ar[-2][0], tr_ar[-1][0]]

        for agent_idx, agent in enumerate(agents):
            if len(agent.track_queue) == 0:
                next_s = agent.mdp.sample(agent.state, simulation.ani.event)
                agent.track_queue += track_outs(
                    (Util.prod2state(agent.state, agent.states), Util.prod2state(next_s, agent.states)))
                if agent_idx == 0:
                    simulation.pause_flag = True
                    simulation.time_step += 1
                    simulation.counter_text.set_text('{}'.format(simulation.time_step))
                    write_objects += [simulation.counter_text]
                if agent.track_queue[0] in simulation.observable_states:
                    agent.update_value(simulation.ani.event, next_s)
                    print('{} at {}: {}'.format(agent_idx,simulation.time_step,agent.max_delta))
                    write_objects += agent.update_belief(agent.belief_values, -2)
                    agent.highlight_reel.add_item(time_step=simulation.time_step, max_delta=agent.max_delta,
                                                  prev_state=Util.prod2state(agent.state, agent.states),
                                                  next_state=Util.prod2state(next_s, agent.states),
                                                  trigger=simulation.ani.event)
                agent.state = next_s
                agent.dis  = Util.prod2dis(agent.state,agent.states)
            non_write_dis = [0,1,2]
            non_write_dis.pop(Util.prod2dis(agent.state, agent.states))
            write_objects += [agent.c_set[c_j].set_visible(False) for c_j in non_write_dis]
            c_i = agent.c_set[Util.prod2dis(agent.state, agent.states)]
            c_i.set_visible(True)
            b_i = agent.b_i
            agent.activate_belief_plt()
            b_i.set_visible(True)
            text_i = agent.t_i
            if simulation.ani.moving:
                agent_pos = agent.track_queue.pop(0)
            else:
                agent_pos = agent.track_queue[0]
            loc = tuple(reversed(Util.coords(agent_pos - 30, ncols)))
            # Use below line if you're working with circles
            b_i.set_center([loc[0] + 1, loc[1] - 1])

            # Use this line if you're working with images
            c_i.xy = loc
            c_i.xyann = loc
            c_i.xybox = loc

            if agent_pos in building_squares:
                c_i.offsetbox.image.set_alpha(0.35)
            else:
                c_i.offsetbox.image.set_alpha(1.0)

            agent.c_set[Util.prod2dis(agent.state, agent.states)] = c_i
            if agent.belief > 0.75:
                b_i.set_visible(True)
            else:
                b_i.set_visible(False)
            write_objects += [c_i, b_i]

        for r_i in rl_ar:
            r_i[0].set_visible(False)
        # print([Util.prod2dis(i.state, i.states) for i in agents])
        rl_ar[Util.prod2dis(agents[0].state, agents[0].states)][0].set_visible(False)
        write_objects += rl_ar

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
            write_objects += [c_i, b_i]

        # Update everything
        SimulationRunner.instance = simulation
        SimulationRunner.instance.tr_ar = tr_ar

        return write_objects

    def on_press(event):
        ani = SimulationRunner.instance.ani
        sim_inst = SimulationRunner.instance

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
            SimulationRunner.instance.add_agent()
        elif event.key.lower() == "-":  # remove agent
            SimulationRunner.instance.remove_agent()
        elif event.key.lower() == "m":
            ani.moving ^= True
        elif event.key.lower() == "r":
            # Up and Down the road
            sim_inst.ani.sensor_state = SensorState.ADD_SENSOR
            sim_inst.sensors.append(Sensor((15, 21),
                                           observable_regions=SimulationRunner.instance.observable_regions,
                                           ncols=SimulationRunner.instance.ncols, ax=SimulationRunner.instance.ax,
                                           moving=deque(list(range(645, 655)) + list(reversed(range(645, 655))) + list(
                                               reversed(range(635, 645))) + list(range(635, 645))),
                                           gwg=SimulationRunner.instance.gwg))

        # update simulation animation
        SimulationRunner.instance.ani = ani

    def on_click(event):
        sim_inst = SimulationRunner.instance
        if event.button == 1:
            sim_inst.ani.sensor_state = SensorState.ADD_SENSOR
            sim_inst.sensors.append(Sensor(tuple(reversed((floor(event.xdata), floor(event.ydata)))),
                                                            observable_regions=SimulationRunner.instance.observable_regions,
                                                            ncols=SimulationRunner.instance.ncols,ax=SimulationRunner.instance.ax,gwg=SimulationRunner.instance.gwg))
        if event.button == 2:
            ## Random moving
            sim_inst.ani.sensor_state = SensorState.ADD_SENSOR
            sim_inst.sensors.append(Sensor(tuple(reversed((floor(event.xdata), floor(event.ydata)))),
                                                            observable_regions=SimulationRunner.instance.observable_regions,
                                                            ncols=SimulationRunner.instance.ncols,ax=SimulationRunner.instance.ax,moving=True,gwg=SimulationRunner.instance.gwg))
        elif event.button == 3:
            sim_inst.ani.sensor_state = SensorState.REMOVE_SENSOR
            sense_dists = [s_i.dist_to_sense(tuple((floor(event.xdata), floor(event.ydata)))) for s_i in sim_inst.sensors]
            del_ind = np.argmin(sense_dists)
            del_sen = sim_inst.sensors.pop(del_ind)
            del del_sen


            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))

        # update simulation animation
        SimulationRunner.instance = sim_inst

def run_interactive_sim(agent_indices, event_names, agent_track_queues=None):
    """
    Runs an interactive simulation showing the plt gridworld
    for agents with indices `agent_indices` and events
    and their values mapped in `event_names`.

    `agent_track_queues` can be an array of tracks to send into grid_init
    if pre-loaded tracks want to be run. if None, then the track queues start
    out empty for each agent (default)
    """
    mdp_list, mdp_states, mdp_keys = ERSA_Env()
    mc_dict = dict()
    env_states = list(mdp_states.keys())
    for a in event_names.keys():
        mc_dict.update({a: [m.construct_MC(dict([[s, [a]] for s in env_states])) for m in mdp_list]})

    mc_dict = dict()
    for a in event_names.keys():
        mc_dict.update({a: [m.construct_MC(dict([[s, [a]] for s in env_states])) for m in mdp_list]})

    Simulation.mc_dict = mc_dict

    with open('VisibleStates.json') as json_file:
        observable_regions = json.load(json_file)

    my_dpi = 350
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=2.0, metadata=dict(artist='Me'), bitrate=1800)
    fig = plt.figure(figsize=(3600 / my_dpi, 2000 / my_dpi), dpi=my_dpi)

    # ---------- PART 2:
    nrows = 30
    ncols = 30
    regionkeys = {'pavement', 'gravel', 'grass', 'sand', 'deterministic'}
    regions = dict.fromkeys(regionkeys, {-1})
    regions['deterministic'] = range(nrows * ncols)

    agents, tr_ar, rl_ar, building_squares, ax, counter_text = SimulationRunner.grid_init(nrows=nrows, ncols=ncols,
                                                                                          desiredIndices=agent_indices,
                                                                                          agent_track_queues=agent_track_queues)
    gwg = Gridworld([0], nrows=nrows, ncols=ncols, regions=regions, obstacles=building_squares)
    fig.canvas.mpl_connect('key_press_event', SimulationRunner.on_press)
    fig.canvas.mpl_connect('button_press_event', SimulationRunner.on_click)
    # ani = FuncAnimation(fig, update_all, frames=10, interval=1250, blit=True, repeat=True)
    anim = FuncAnimation(fig, SimulationRunner.update_all, frames=frames, interval=1, blit=False, repeat=False)
    SimulationRunner(anim, gwg, ax, agents, tr_ar, rl_ar, observable_regions, building_squares, nrows, ncols,
                     counter_text)
    # anim.save('Environment-Slide3_Video.mp4',fps=24, extra_args=['-vcodec', 'libx264'])
    plt.show()

    # can return more things later on if needed
    return agents, anim

def main():
    # ---------- PART 1:
    # grab arguments of format: 'Store A Owner','Store B Owner','Repairman','Shopper','Suscipious','Home Owner'
    agent_indices = Util.get_agent_indices(sys.argv)
    event_names = {0: 'nominal', 1: 'iceA', 2: 'iceB', 3: 'iceC', 4: 'alarmA', 5: 'alarmB', 6: 'alarmG'}

    # run initial interactive simulation with command line initial locations
    agents, anim = run_interactive_sim(agent_indices=agent_indices, event_names=event_names)

    # ---------- PART 3:
    # code to show highlights for a specific agent (executed after a run)
    # for now, just take an idx from commandline.
    chosen_agent_idx = int(input("Enter agent idx to display highlights for: "))
    chosen_agent = None
    for agent in agents:
        if agent.agent_idx == chosen_agent_idx:
            chosen_agent = agent
            break
    # load the most significant highlight data for the agent
    highlights = chosen_agent.highlight_reel.get_items()
    # for every highlight, run an animation
    for i in range(len(highlights)):
        print(f"running highlight {i}: {highlights[i]}")
        prev_state = int(chosen_agent.highlight_reel.get_item_value(i, "prev_state"))
        next_state = int(chosen_agent.highlight_reel.get_item_value(i, "next_state"))
        # print(f"from {prev_state} to {next_state}:")

        track_queue = track_outs((prev_state, next_state))
        # print(track_queue)

        # form agent_indices just for the 1 agent
        highlight_agent_indices = [(chosen_agent_idx, prev_state)]
        # print(highlight_agent_indices)

        # reset animation stuff so things run
        SimulationRunner.instance = None
        del anim

        # run only the track
        # TODO: haven't got to the part where it stops running the animation after the track ends
        # TODO: need to also show the trigger for each highlight
        _, anim = run_interactive_sim(agent_indices=highlight_agent_indices, event_names=event_names,
                                      agent_track_queues=[track_queue])


# anim.save('Environment-Slide3_Video.mp4',writer=writer)


if __name__ == '__main__':
    main()
