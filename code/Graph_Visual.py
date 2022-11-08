import matplotlib
from Graph import Graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np


matplotlib.use('Qt5Agg')

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
            self.loc_dict.update({self.agents[col]: loc})
            p_c = patches.Circle(loc, radius/8, color=my_palette(col), zorder=4)
            self.ax.add_artist(p_c)
            cir_array.append(p_c)
        self.line_dict = {}
        for l_d in self.loc_dict:
            for k_d in self.loc_dict:
                x_point, y_point = zip(*[self.loc_dict[l_d], self.loc_dict[k_d]])
                p_l, = self.ax.plot(x_point, y_point, linewidth=5, color=(0, 0, 0), zorder=1)
                p_l.set_visible(False)
                self.line_dict.update({(l_d, k_d): p_l})
    def update_graph(self):
        for l_i in self.line_dict.values():
            l_i.set(visible=False)
        for a_i in self.vertices.keys():
            for a_j in self.vertices[a_i]:
                self.line_dict[(a_i,a_j)].set(visible=True)
