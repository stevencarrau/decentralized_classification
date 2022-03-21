import os

class Util:
    @staticmethod
    def prod2state(s_in, prod_keys):
        return prod_keys[s_in][0]
    @staticmethod
    def prod2stateSet(set_state,prod_keys):
        return set([prod_keys[s_in][0] for s_in in set_state])

    @staticmethod
    def prod2dis(s_in, prod_keys):
        return prod_keys[s_in][1]

    @staticmethod
    def prod2disSet(set_state, prod_keys):
        return set([prod_keys[s_in][1] for s_in in set_state])


    @staticmethod
    def state2prod(s_in, dis_in, states):
        return states[(s_in, dis_in)]

    @staticmethod
    def state2prodSet(set_pairs,states):
        return set([states[(s_in[0],s_in[1])]for s_in in set_pairs])

    @staticmethod
    def coords(s, ncols):
        return int(s / ncols), int(s % ncols)

    @staticmethod
    def coord2state(coords, ncols):
        return int(coords[0] * ncols) + int(coords[1])

    @staticmethod
    def track2coords(track, ncols):
        """Given a track and the number of columns on the board,
        return the x,y coordinates representing the track.
        """
        return [tuple(reversed(Util.coords(track[i] - 30, ncols))) for i in range(len(track))]

    @staticmethod
    def prepare_dir(dir):
        """Either create the directory if it's not a thing already.
        If the directory is already created, clean it."""
        if not (os.path.exists(dir) and os.path.isdir(dir)):
            # make a new one from scratch
            os.makedirs(dir)
        else:
            # up to the user to delete old folder contents, may be
            # unsafe if a 'delete directory' is inserted here to clean
            # the previous contents
            pass

    @staticmethod
    def get_agent_indices(args):
        numAgents = len(args) - 1
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

        return sorted(inputs)

from collections import OrderedDict

# class LimitedSizeDict(OrderedDict):
#     def __init__(self, *args, **kwds):
#         self.size_limit = kwds.pop("size_limit", None)
#         OrderedDict.__init__(self, *args, **kwds)
#         self._check_size_limit()
#
#     def __setitem__(self, key, value):
#         OrderedDict.__setitem__(self, key, value)
#         self._check_size_limit()
#
#     def _check_size_limit(self):
#         if self.size_limit is not None:
#             while len(self) > self.size_limit:
#                 self.popitem(last=False)
#     def update(self, __m: Mapping[_KT, _VT], **kwargs: _VT) -> None:
