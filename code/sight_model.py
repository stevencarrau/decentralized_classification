import numpy as np
import shapely.geometry
import json_writer
from gridworld import *
import itertools
# import ray
# import tqdm


# @ray.remote
def invis(gw, state,visdist):
    targcoords = list(gw.coords(state))
    targcoords[0] += 0.5
    targcoords[1] += 0.5
    invisstates = set()
    for s in gw.states:
        statecoords = list(gw.coords(s))
        statecoords[0] += 0.5
        statecoords[1] += 0.5
        line = shapely.geometry.LineString([targcoords, statecoords])
        dist = np.sqrt((targcoords[0]-statecoords[0])**2 + (targcoords[1]-statecoords[1])**2)
        if dist <= visdist: # if target is not too far
            for obs in gw.obstacles:
                obscoordsupleft = list(gw.coords(obs))
                obscoordsupright = [obscoordsupleft[0] + 0.99, obscoordsupleft[1]]
                obscoordsbotleft = [obscoordsupleft[0], obscoordsupleft[1] + 0.99]
                obscoordsbotright = [obscoordsupleft[0] + 0.99, obscoordsupleft[1] + 0.99]
                obshape = shapely.geometry.Polygon([obscoordsupleft, obscoordsupright, obscoordsbotright, obscoordsbotleft])
                isVis = not line.intersects(obshape)
                if not isVis:
                    invisstates.add(s)
                    break

        else:
            invisstates.add(s)
    invisstates = invisstates - set(gw.obstacles)
    return frozenset(invisstates)

#
# def isVis(gw, state, target):
#     invisstates = invis(gw, state,10)
#     visstates = set(gw.states) - invisstates - set(gw.obstacles)
#     if target in visstates:
#         return True
#     else:
#         return False


nrows = 30
ncols = 30

regionkeys = {'pavement', 'gravel', 'grass', 'sand', 'deterministic'}
regions = dict.fromkeys(regionkeys, {-1})
regions['deterministic'] = range(nrows * ncols)


# ray.init()
storeA_squares = [list(range(m, m + 5)) for m in range(366, 486, 30)]
home_squares = [list(range(m, m + 5)) for m in range(380, 500, 30)]
storeB_squares = [list(range(m, m + 5)) for m in range(823, 890, 30)]
building_squares = list(itertools.chain(*home_squares)) + list(itertools.chain(*storeA_squares)) + list(itertools.chain(*storeB_squares))
gwg = Gridworld([0], nrows=nrows, ncols=ncols,regions=regions,obstacles=building_squares)
observable_regions = {i:list(set(gwg.states) - invis(gwg,i,10)-set(gwg.obstacles)) for i in gwg.states}
    #
    # invis_list = [invis.remote(gwg, i, j) ]
    # invis_out = ray.get(invis_list)
    # observable_regions.update({i:[ind_j for ind_j,j in enumerate(invis_out) if j]})
json_writer.write_JSON('VisibleStates.json',observable_regions)