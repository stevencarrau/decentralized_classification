from gridworld import *
from mdp import MDP
from pomdp import POMDP
from tqdm import tqdm
nrows = 7
ncols = 5
goodtargets = {0}
badtargets = {20}
initial = [4]
moveobstacles = [30]
targets = [list(goodtargets.union(badtargets))]
obstacles = []


regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['sand']= range(nrows*ncols)

gwg = Gridworld(initial, nrows, ncols, 1, targets, obstacles,moveobstacles,regions)
gwg.render()
gwg.draw_state_labels()
gwg.save('Examples/example_7x5.png')
#
states = range(gwg.nstates)
alphabet = [0,1,2,3] # North, south, west, east
transitions = []
for s in states:
    for a in alphabet:
        for t in np.nonzero(gwg.prob[gwg.actlist[a]][s])[0]:
            p = gwg.prob[gwg.actlist[a]][s][t]
            transitions.append((s, alphabet.index(a), t, p))

mdp = MDP(states, set(alphabet),transitions)

# V, goodpolicy = mdp.max_reach_prob(goodtargets, epsilon=0.0001)
# V, badpolicy = mdp.max_reach_prob(badtargets, epsilon=0.0001)
randomness = 0
R = dict([(s,a,next_s),0.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s,a) )
R.update([(s,a,next_s),1.0] for s in mdp.states  for a in mdp.available(s) for next_s in mdp.post(s,a) if next_s in goodtargets and s in goodtargets)
V,goodpolicy =  mdp.T_step_value_iteration(R,10)
R = dict([(s,a,next_s),0.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s,a) )
R.update([(s,a,next_s),1.0] for s in mdp.states  for a in mdp.available(s) for next_s in mdp.post(s,a) if next_s in badtargets and s in badtargets)

V,badpolicy =  mdp.T_step_value_iteration(R,10)
good_MC = mdp.construct_MC(goodpolicy,'Examples/7x5_good.txt')
bad_MC = mdp.construct_MC(badpolicy,'Examples/7x5_bad.txt')

# Construct product mdp
states = [(s1,s2) for s1 in gwg.states for s2 in gwg.states]
product_trans = []
for s1 in states:
    for s2 in states:
        for a in alphabet:
            p1 = gwg.prob[gwg.actlist[a]][s1[0]][s2[0]]
            p2 = bad_MC[(s1[1],s2[1])]
            if p1*p2>0:
                product_trans.append((s1,a,s2,p1*p2))

product_mdp = MDP(states, set(alphabet),product_trans)
product_pomdp = POMDP(product_mdp,gwg)
product_mdp.write_to_file('Examples/7x5_productmdp_bad',(30,4))
product_pomdp.write_to_file('Examples/7x5_productpomdp_bad',(30,4))

# Construct product mdp
states = [(s1,s2) for s1 in gwg.states for s2 in gwg.states]
product_trans2 = []
for s1 in states:
    for s2 in states:
        for a in alphabet:
            p1 = gwg.prob[gwg.actlist[a]][s1[0]][s2[0]]
            p2 = good_MC[(s1[1],s2[1])]
            if p1*p2>0:
                product_trans2.append((s1,a,s2,p1*p2))

product_mdp2 = MDP(states, set(alphabet),product_trans2)
product_pomdp2 = POMDP(product_mdp2,gwg)
product_mdp2.write_to_file('Examples/7x5_productmdp_good',(30,4))
product_pomdp2.write_to_file('Examples/7x5_productpomdp_good',(30,4))

