
class Util:
    @staticmethod
    def prod2state(s_in, prod_keys):
        return prod_keys[s_in][0]

    @staticmethod
    def prod2dis(s_in, prod_keys):
        return prod_keys[s_in][1]

    @staticmethod
    def state2prod(s_in, dis_in, states):
        return states[(s_in, dis_in)]

    @staticmethod
    def coords(s, ncols):
        return int(s / ncols), int(s % ncols)

    @staticmethod
    def coord2state(coords, ncols):
        return int(coords[0] * ncols) + int(coords[1])

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
