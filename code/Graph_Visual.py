import matplotlib
from Graph import Graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap
c = ["lightcoral","white", "palegreen"]
v = [0,.5,1.]
l = list(zip(v,c))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

# matplotlib.use('Qt5Agg')

class GraphVisual(Graph):
    def draw_graph(self,fig_dim):
        fig = plt.figure(figsize=(fig_dim,fig_dim))
        no_agents = len(self.agents)
        angles = [n / float(no_agents) * 2 * math.pi for n in range(no_agents)]
        angles += angles[:1]
        my_palette = plt.cm.get_cmap("tab10", no_agents)
        self.ax = plt.subplot(111)
        plt.axis('off')
        self.ax.set_xlim([-self.ax.get_window_extent().height / 2, self.ax.get_window_extent().height / 2])
        self.ax.set_ylim([-self.ax.get_window_extent().height / 2, self.ax.get_window_extent().height / 2])
        radius = self.ax.get_window_extent().height / 2 - 50
        cir_array = []
        self. loc_dict = {}
        for col, a_i in enumerate(angles[:-1]):
            loc = tuple(np.array([radius * math.sin(a_i), radius * math.cos(a_i)]))
            self.loc_dict.update({self.agents[col].name: loc})
            p_c = patches.Circle(loc, radius/8, zorder=4,edgecolor='k')
            # self.ax.add_artist(p_c)
            cir_array.append(p_c)
        p = PatchCollection(cir_array,cmap=cmap,match_original=True)
        p.set_array(np.random.random(5))
        self.circles = self.ax.add_collection(p)
        self.line_dict = {}
        for l_d in self.loc_dict:
            for k_d in self.loc_dict:
                if l_d != k_d:
                    x_point, y_point = zip(*[self.loc_dict[l_d], self.loc_dict[k_d]])
                    p_l, = self.ax.plot(x_point, y_point, linewidth=5, color=(0, 0, 0), zorder=1)
                    p_l.set_visible(False)
                    self.line_dict.update({(l_d, k_d): p_l})
    def update_graph(self,belief_array=None):
        for l_i in self.line_dict.values():
            l_i.set(visible=False)
        for a_i in self.vertices.keys():
            for a_j in self.vertices[a_i]:
                self.line_dict[(a_i.name,a_j.name)].set(visible=True)
        if belief_array:
            self.circles.set_array(np.array(belief_array))
            # self.circles.set_array(np.random.random(5))