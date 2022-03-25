from live_sim import *
import numpy as np
from util import Util
import sys
import matplotlib.image as mpimg
import networkx as nx

Util.assert_livesim_data_exists(agents_save_path)

network_images_save_path = "network_images"

def main():
    """
    Visually show the relation between different agents as a graph.
    Need to enter in the agent_idx for which agent to show relations
    for as a cmdline argument input
        -> Usage: `python network_graph.py AGENT_IDX`
    """
    agents = np.load(agents_full_fpath, allow_pickle=True).tolist()
    assert len(sys.argv) == 2, "Must specify the agent idx through program arguments"
    chosen_agent_idx = int(sys.argv[1])
    print("chosen agent idx:", chosen_agent_idx)
    chosen_agent = get_agent_with_idx(chosen_agent_idx, agents)
    other_agents = [agent for agent in agents if agent.agent_idx != chosen_agent_idx]

    network = nx.Graph()
    # add all nodes first
    for agent in agents:
        idx = agent.agent_idx
        img_path = agent_image_paths[idx] + ".png"
        img = mpimg.imread(img_path)
        network.add_node(idx, image=img)
    nodes = network.nodes()
    print("nodes:", network.nodes())

    # add the edges, weights, and images now
    for other_agent in other_agents:
        other_agent_idx = other_agent.agent_idx
        # weight depends on how many times `other_agent` was in a neighbor state
        # of `chosen_agent`
        edge_weight = chosen_agent.agent_in_neighbor_state_freqs.get(other_agent_idx, 0)
        network.add_edge(chosen_agent_idx, other_agent_idx, weight=edge_weight)
    print("edges:", network.edges())

    pos = nx.spring_layout(network) #nx.circular_layout(network)
    edges = network.edges()
    weights = [network[u][v]['weight'] for u, v in edges]

    my_dpi = 350
    fig = plt.figure(figsize=(3600 / my_dpi, 2000 / my_dpi), dpi=my_dpi)
    ax = plt.subplot(111)
    # ax.set_aspect('equal')
    # draw nodes (agents)
    nx.draw_networkx_nodes(network, pos, nodelist=nodes, ax=ax, node_color='none')
    # draw edges
    nx.draw_networkx_edges(network, pos, edgelist=edges, ax=ax)
    # draw edge labels
    weight_labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, pos=pos, edge_labels=weight_labels, ax=ax)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    for n in network:
        # make chosen agent look bigger than others
        if n == chosen_agent_idx:
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


    # save the image as a picture
    Util.prepare_dir(network_images_save_path)
    agent_character_names = ['Captain_America', 'Black_Widow', 'Hulk', 'Thor', 'Thanos', 'Ironman']
    agent_name = agent_character_names[chosen_agent_idx]
    network_image_full_fpath = f"{network_images_save_path}/{agent_name}_network.png"
    print(f"\nSaving image to {network_image_full_fpath}...")
    plt.savefig(network_image_full_fpath)
    # plt.close()
    print("Finished saving image.")

if __name__ == '__main__':
    main()