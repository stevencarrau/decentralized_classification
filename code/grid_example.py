from gridworld import *
from mdp import *
from policy import Policy
from agent import Agent

def play_sim(multicolor=True, policy=None,grid=None):
    agent_loc = dict([[a.id_no, a.current] for a in policy])
    # Initialize missing status
    for a_a in policy:
        a_a.initLastSeen(agent_loc.keys(),agent_loc.values())
    target_union = set()
    for t in grid.targets:
        target_union.update(set(t))
    while any([grid.current[i][0] not in target_union for i in range(grid.nagents)]):
        # nom_policy = []
        for idx_j, j in enumerate(grid.current):
            if policy is None:
                while True:
                    arrow = grid.getkeyinput()
                    if arrow != None:
                        break
                grid.current[idx_j] = int(
                    np.random.choice(range(grid.prob[arrow][grid.current[idx_j]].reshape(-1, ).shape[0]), None, False,
                                     grid.prob[arrow][grid.current[idx_j]].reshape(-1, )))
            else:
                arrow = grid.actlist[policy[idx_j].policy.sample(j)]
                s, q = grid.current[idx_j]
                prev_state = grid.current[idx_j]
                # pygame.time.wait(50)
                s = int(np.random.choice(range(grid.prob[arrow][s].reshape(-1, ).shape[0]), None, False,
                                         grid.prob[arrow][s].reshape(-1, )))
                if s == grid.targets[idx_j][q]:
                    q += 1
                    if q == len(grid.targets[idx_j]):
                        q = 0
                policy[idx_j].updateAgent((s,q))
                agent_loc[policy[idx_j].id_no] = policy[idx_j].current
                grid.current[idx_j] = (s, q)
                grid.agent_list[idx_j].updatePosition(grid.indx2coord(grid.current[idx_j][0], center=True),
                                                      grid.obsbox(grid.current[idx_j][0],
                                                                  grid.agent_list[idx_j].obs_range))
                policy[idx_j].policy.updateNominal(grid.current[idx_j])
                grid.agent_list[idx_j].updateRoute([list(reversed(grid.indx2coord(r_i[0], center=True))) for r_i in
                                                    policy[idx_j].policy.nom_trace.values()])
                print("Local likelihood for ", idx_j, ": ",
                      policy[idx_j].policy.observation(grid.current[idx_j], [prev_state], 1))
        for p_i in policy:
            p_i.updateVision(p_i.current,agent_loc)
        grid.render(multicolor=multicolor, nom_policy=True)
        pygame.time.wait(1000)
    pygame.quit()
    return print("Goal!")


# from tqdm import tqdm
nrows = 10
ncols = 10
initial = [(33,0),(41,0),(7,0),(80,0)]
moveobstacles = []
targets = [[0,9],[60,69],[20,39],[79,95]]
public_targets = [[0,9],[60,69],[20,39],[55,95]]
obstacles = []
obs_range = 3

evil_switch = True

regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['pavement']= range(nrows*ncols)
regions_det = dict.fromkeys(regionkeys,{-1})
regions_det['deterministic'] = range(nrows*ncols)

gwg = Gridworld(initial, nrows, ncols, len(initial), targets, obstacles,moveobstacles,regions,public_targets=public_targets)
det_gw = Gridworld(initial, nrows, ncols, len(initial), targets, obstacles,moveobstacles,regions_det)
gwg.render(multicolor=True)
# gwg.draw_state_labels()
# gwg.save('Examples/example_7x5.png')
#
states = range(gwg.nstates)
alphabet = [0,1,2,3] # North, south, west, east
transitions = []
det_trans = []
for s in states:
    for a in alphabet:
        for t in np.nonzero(gwg.prob[gwg.actlist[a]][s])[0]:
            p = gwg.prob[gwg.actlist[a]][s][t]
            transitions.append((s, alphabet.index(a), t, p))
        for t2 in np.nonzero(det_gw.prob[det_gw.actlist[a]][s])[0]:
            p_det = det_gw.prob[det_gw.actlist[a]][s][t2]
            det_trans.append((s, alphabet.index(a), t2, p_det))

mdp = MDP(states, set(alphabet),transitions)
nfa = MDP(states,set(alphabet),det_trans) #deterministic transitions
print("Models built")
agent_array = []
c_i = 0
print("Computing policies")
for i,j,k in zip(initial,targets,public_targets):
    if evil_switch:
        agent_array.append(Agent(i,j,k,mdp,nfa,gwg))
    else:
        agent_array.append(Agent(i,k, k, mdp, nfa,gwg))
    print("Policy ",c_i," -- complete")
    c_i += 1



play_sim(True,agent_array,gwg)


