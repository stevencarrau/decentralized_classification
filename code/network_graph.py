from live_sim import *
import numpy as np
from util import Util
import sys
import matplotlib.image as mpimg
import networkx as nx
from itertools import permutations
from matplotlib.lines import Line2D

Util.assert_livesim_data_exists(agents_save_path)

# where the images will be saved to
network_images_save_path = "network_images"

# the threshold for the edge weight to show up
freq_threshold_percentage = 0.20

def plt_draw_network(network, pos, situation, situation_timesteps, agent_idxs_to_amplify=None, show=False):
    """
    Draws the network on a plt Canvas.
    network: a networkx graph
    pos: a networkx graph layout
    situation: a string representing the trigger or situation to show data for
    situation_timesteps: the number of timesteps spent in `situation`
    agent_centric_idxs: if specified, makes these agents look bigger
    show: if True, does plt.show(). Warning: must set to False if you want
        to save to a picture. Or, add code to save to a picture before plt.show().
    """
    if agent_idxs_to_amplify is None:
        agent_idxs_to_amplify = []

    nodes = network.nodes()
    edges = network.edges()

    # have a color map gradient so colors indicate edge weight
    rgba = lambda r, g, b, a: (r/255, g/255, b/255, a)
    gradient_map = [
        rgba(208, 222, 33, 1),    # yellow
        # rgba(255, 255, 0, 1),        # highlighter yellow
        rgba(79, 220, 74, 1),       # green
        # rgba(63, 218, 216, 1),    # cyan
        # rgba(47, 201, 226, 1),      # darker cyan
        rgba(28, 127, 238, 1),    # blue
        # rgba(95, 21, 242, 1),       # purple
        # rgba(186, 12, 248, 1),    # dark pink
        # rgba(251, 7, 217, 1),       # light pink
        rgba(255, 154, 0, 1),       # orange
        rgba(255, 0, 0, 1)        # red
    ]
    gradient_map_bin_size = 100 // len(gradient_map)
    # '0-20%', '20-40%' etc
    gradient_labels = [f"{gradient_map_bin_size * i}-{gradient_map_bin_size * (i + 1)}%" for i in
                       range(len(gradient_map))]

    edge_coloring = []
    for u,v in edges:
        # put the weight into 1 of the bins above
        frac = network[u][v]['weight']
        bin = round(frac * 100) // gradient_map_bin_size
        # in the case that frac is actually 1, we need to adjust index logic
        if bin == len(gradient_map):
            bin -= 1
        color = gradient_map[bin]
        edge_coloring.append(color)

    # uniform thickness for all edges
    thicknesses = [3 for u,v in edges]

    my_dpi = 350
    fig = plt.figure(figsize=(3600 / my_dpi, 2000 / my_dpi), dpi=my_dpi)
    ax = plt.subplot(111)
    # ax.set_aspect('equal')
    # draw edges
    ec = nx.draw_networkx_edges(network, pos, ax=ax, width=thicknesses, edge_color=edge_coloring)
    # draw nodes (agents)
    nc = nx.draw_networkx_nodes(network, pos, ax=ax, node_color='none')
    # draw edge labels: don't do this because it'll get busy very quickly
    # weight_labels = nx.get_edge_attributes(network, 'weight')
    # nx.draw_networkx_edge_labels(network, pos=pos, edge_labels=weight_labels, ax=ax)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    for n in network:
        # make agent look bigger if specified
        if n in agent_idxs_to_amplify:
            imgsize = 0.15
        else:
            imgsize = 0.1
        p2 = imgsize / 2.0
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes([xa - p2, ya - p2, imgsize, imgsize])
        # a.set_aspect('equal')
        a.imshow(network.nodes[n]['image'])
        a.axis('off')
    ax.axis('off')

    # draw the color legend
    handles = [Line2D([], [], color=color, label=label)
               for color, label in zip(gradient_map, gradient_labels)]
    ax.legend(handles=handles, loc="upper center", ncol=len(gradient_map), fontsize="small",
              title=f"Percentage Spent in Neighboring States out of {situation_timesteps} timesteps for '{situation}' "
                    f"trigger")

    if show:
        plt.show()

def plt_save_picture(picture_title):
    """
    Saves what is currently on the plt canvas to a picture on
    disk with name picture_title. Note, this must be called
    before plt.show() in order to actually save.
    """
    Util.prepare_dir(network_images_save_path)
    network_image_full_fpath = f"{network_images_save_path}/{picture_title}_network.png"
    print(f"\nSaving image to '{network_image_full_fpath}'...")
    plt.savefig(network_image_full_fpath)
    print("Finished saving image.")

def single_agent_centric_graph(agents, agent_idx, situation_filter):
    """
    Visually show the relation between different agents as a graph.
    agents: numpy array of agents
    agent_idx: idx to show graph for
    situation_filter: which situations to show the network graph for.
    """
    chosen_agent = get_agent_with_idx(agent_idx, agents)
    other_agents = [agent for agent in agents if agent.agent_idx != agent_idx]

    assert situation_filter in Agent.situation_type_to_events, "Situation type must be one of the situations specified" \
                                                               " in Agent.situation_type_to_events: " \
                                                               f"{Agent.situation_type_to_events.keys()}"

    # get the total number of timesteps that the agent spent in that trigger overall
    total_timesteps = chosen_agent.neighbor_state_freqs[situation_filter]["total_timesteps"]

    network = nx.Graph()
    # add all nodes first
    for agent in agents:
        idx = agent.agent_idx
        img_path = agent_image_paths[idx] + ".png"
        img = mpimg.imread(img_path)
        network.add_node(idx, image=img)
    print("nodes:", network.nodes())

    # add the edges, weights, and images now
    for other_agent in other_agents:
        other_agent_idx = other_agent.agent_idx
        # weight depends on how many times `other_agent` was in a neighbor state
        # of `chosen_agent`
        freq = chosen_agent.neighbor_state_freqs[situation_filter].get(other_agent_idx, 0)
        freq = freq / total_timesteps

        # skip if it goes over the threshold
        if freq < freq_threshold_percentage:
            continue
        edge_weight = freq

        network.add_edge(agent_idx, other_agent_idx, weight=edge_weight)
    print("edges:", network.edges())

    # circular layout, but emphasized agent will be larger than others
    pos = nx.circular_layout(network)

    plt_draw_network(network=network, pos=pos, situation=situation_filter, situation_timesteps=total_timesteps,
                     agent_idxs_to_amplify=[agent_idx])

    # save the image as a picture
    agent_character_names = ['Captain_America', 'Black_Widow', 'Hulk', 'Thor', 'Thanos', 'Ironman']
    agent_name = agent_character_names[agent_idx]
    plt_save_picture(picture_title=f"{agent_name}_{situation_filter}")

def all_agents_graph(agents, situation_filter):
    """
    Shows the relationships between all agents on a single graph.
    As opposed to single_agent_centric_graph(), which shows the
    frequencies of other agents entering neighboring regions
    in relation to a single centric agent.

    Shows all agents as vertices. An edge appears between 2 agents
    if they appeared in neighboring states.
    The weight of the edge is a fraction: the number of timesteps
    where the agents appeared in neighboring states over the
    total number of timesteps in the entire episode.

    agents: numpy array, should be loaded from disk from a live_sim run
    situation_filter: which situations to show the network graph for.
    """
    assert situation_filter in Agent.situation_type_to_events, "Situation type must be one of the situations specified"\
                                                                " in Agent.situation_type_to_events: " \
                                                               f"{Agent.situation_type_to_events.keys()}"

    # assert that all the timesteps among all agents should be equal for this situation
    all_timesteps = [agent.neighbor_state_freqs[situation_filter]["total_timesteps"] for agent in agents]
    total_timesteps = all_timesteps[0]
    assert all(val == total_timesteps for val in all_timesteps), f"timesteps for situation {situation_filter} should " \
                                                                 f"be equal for all agents"

    network = nx.Graph()
    # add all nodes first
    for agent in agents:
        idx = agent.agent_idx
        img_path = agent_image_paths[idx] + ".png"
        img = mpimg.imread(img_path)
        network.add_node(idx, image=img)
    print("nodes:", network.nodes())

    # add edges now. ideally this is a complete graph with an edge
    # between every 2 nodes which is why we use permutations to
    # generate all possible size-2 pairs, but some edges may not
    # show up if the stored frequency doesn't pass the threshold
    all_possible_edges = list(permutations(agents, r=2))
    for possible_edge in all_possible_edges:
        agent_u, agent_v = possible_edge
        agent_u_idx = agent_u.agent_idx
        agent_v_idx = agent_v.agent_idx

        # get the stored frequencies from the dicts, relative to each other
        agent_u_centric_freq = agent_u.neighbor_state_freqs[situation_filter].get(agent_v_idx, 0)
        agent_v_centric_freq = agent_v.neighbor_state_freqs[situation_filter].get(agent_u_idx, 0)

        # they must be the same: it's like they are within radius of each other so
        # live_sim should have realized that and say both are neighbors to each other
        assert agent_u_centric_freq == agent_v_centric_freq

        # now we can get the edge weight since the freqs are the same from both ends
        freq = agent_u_centric_freq
        freq_fraction = freq / total_timesteps

        # skip if it doesn't meet the threshold
        if freq_fraction < freq_threshold_percentage:
            continue
        # otherwise, the edge qualifies: add the edge
        network.add_edge(agent_u_idx, agent_v_idx, weight=freq_fraction)
    print("edges:", network.edges())

    # circular to have all the agents in a circle
    pos = nx.circular_layout(network)

    plt_draw_network(network=network, pos=pos, situation=situation_filter, situation_timesteps=total_timesteps)

    plt_save_picture(picture_title=f"All_Agents_{situation_filter}")



def main():
    agents = np.load(agents_full_fpath, allow_pickle=True).tolist()
    for agent in agents:
        print("agent idx:", agent.agent_idx)
        print("stored data:", agent.neighbor_state_freqs)

    # to save network graphs for all situations, for all agents individually
    # for agent in agents:
    #     for situation in Agent.situation_type_to_events:
    #         single_agent_centric_graph(agents, agent_idx = agent.agent_idx, situation_filter=situation)

    # to save network graphs for all situations, for all agents together
    for situation in Agent.situation_type_to_events:
        all_agents_graph(agents, situation_filter=situation)

if __name__ == '__main__':
    main()