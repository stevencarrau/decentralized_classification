from util import Util
import numpy as np

class Sensor:
    observers = []

    def __init__(self, observer_loc,moving=False,ncols=30,observable_regions = set(range(900)),ax=None,gwg=None):
        self.loc = Util.coord2state(observer_loc,ncols)
        self.ncols = ncols
        self.moving = moving
        self.observable_regions = observable_regions
        self.observable_states = self.observable_regions[str(self.loc)]
        self.ax = ax
        self.sensor_artist = self.add_observer(self.loc)
        if gwg:
            self.gwg = gwg

    def __del__(self):
        self.remove_observer()

    def update_sensor(self):
        if self.moving:
            possible_states = set([np.argmax(self.gwg.prob[i][self.loc]) for i in self.gwg.prob]) - set(self.gwg.obstacles)
            self.loc = int(np.random.choice(list(possible_states)))
            loc = tuple(reversed(Util.coords(self.loc, self.ncols)))
            self.sensor_artist.set_xy(np.array([[loc[0]-0.5,loc[1]-0.5],[loc[0]+0.5,loc[1]-0.5],[loc[0]+0.5,loc[1]+0.5],[loc[0]-0.5,loc[1]+0.5],[loc[0]-0.5,loc[1]-0.5]]))
        self.update_observable_states()
        return self.sensor_artist

    def update_observable_states(self,observer_loc=None):
        if observer_loc:
            self.loc = Util.coord2state(observer_loc,self.ncols)
            self.observable_states = self.observable_regions[str(self.loc)]
        else:
            self.observable_states = self.observable_regions[str(self.loc)]
        return self.observable_states

    def add_observer(self,obs_state):
        # # Sensor.observers.append(obs_state)
        # # Sensor.update_observable_states()
        # write_objects = SimulationRunner.instance.blit_viewable_states()
        o_loc = tuple(reversed(Util.coords(obs_state, self.ncols)))
        if self.moving:
            o_x = self.ax.fill([o_loc[0] - 0.5, o_loc[0] + 0.5, o_loc[0] + 0.5, o_loc[0] - 0.5],
                               [o_loc[1] - 0.5, o_loc[1] - 0.5, o_loc[1] + 0.5, o_loc[1] + 0.5],
                               color='purple', alpha=0.50)[0]
        else:
            o_x = self.ax.fill([o_loc[0] - 0.5, o_loc[0] + 0.5, o_loc[0] + 0.5, o_loc[0] - 0.5],
                                                [o_loc[1] - 0.5, o_loc[1] - 0.5, o_loc[1] + 0.5, o_loc[1] + 0.5],
                                                color='green', alpha=0.50)[0]
        return o_x

    def remove_observer(self):
        if self.sensor_artist:
            self.sensor_artist.remove()

    def dist_to_sense(self,click_site):
        o_loc = tuple(reversed(Util.coords(self.loc, self.ncols)))
        return (o_loc[0]-click_site[0])**2+(o_loc[1]-click_site[1])**2


