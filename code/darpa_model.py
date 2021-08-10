import copy
from itertools import product

from mdp import *


def pathways(state_in, main_path, states_out, act, prob):
    ## Function for describing behavior transiting a path
    # Input is the current_state (state_in), the primary destination (main_path), the possible transitions (states_out), action label (act) and then the probability of statying in the room (prob)
    # Function assigns probability of travelling on its main path and then evenly distributes probs across remaining transitions
    # Output is the list of transitions in (s,a,s',p) form
    list_out = [(state_in, act, main_path, prob)]
    states_out.remove(main_path)
    for s in states_out:
        list_out += [(state_in, act, s, (1 - prob) / len(states_out))]
    return list_out


def inside_room(state_in, main_none, states_out, act, prob):
    ## Function for behavior inside a room
    # Input is the room_state (state_in), the possible transitions (states_out), action label (act) and then the probability of statying in the room (prob)
    # Output assigns probability of staying in the room and then evenly distributes probs across remaining transitions
    list_out = [(state_in, act, state_in, prob)]
    states_out.remove(state_in)
    for s in states_out:
        list_out += [(state_in, act, s, (1 - prob) / len(states_out))]
    return list_out


def perturbed_prob(prob_in, delta, escalation_prob):
    if prob_in == 1.0 or 0.0:
        return prob_in * escalation_prob
    else:
        if prob_in < 0.5:
            return (prob_in + delta) * escalation_prob
        elif prob_in > 0.5:
            return (prob_in - delta) * escalation_prob
        else:
            return prob_in * escalation_prob


# def normalize_trans(list_trans,states,actions):
# 	for mdp in list_trans:
# 		for s in states:
# 			for a in actions:
# 				s_a_list = []

def ERSA_Env():
    ## Action names and length of runtime
    event_triggers = {'nominal': 1, 'ice': 6, 'alarm': 4, 'bang': 2}  # {'Name':execution_time}
    event_names = {0: 'nominal', 1: 'iceA', 2: 'iceB', 3: 'iceC', 4: 'alarmA', 5: 'alarmB',
                   6: 'alarmG'}  # using numbers to label triggers
    event_space = list(range(len(event_triggers)))
    methods = {'pathways': pathways, 'inside_room': inside_room}
    deltas = {0: 0.0, 1: 0.15, 2: 0.35}
    escalation_prob = {0: 0.0, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.5, 5: 0.5, 6: 0.5}

    ## Hard code environment
    # 0 - street, 1 - street outside store A, 2 - store A, 3 - electricity box, 4 - street outside store B, 5 - store B, 6 - street outside home, 7 - home
    env_states = [0, 1, 2, 3, 4, 5, 6, 7]
    disruption_status = list(range(len(deltas)))
    product_space = product(env_states, disruption_status)
    product_states = {i: j for i, j in enumerate(product_space)}
    product_reverse = {product_states[i]: i for i in product_states}

    env_trans = {'pathways': {0: ([0, 1, 4, 6]), 1: ([1, 2, 0]), 4: ([0, 5]), 6: ([0, 6, 7])},
                 'inside_room': {2: ([1, 2, 3]), 3: ([3, 2]), 5: ([4, 5]), 7: ([6, 7])}
                 }
    disrupt_trans = {i - 1: [i - 1, i] for i in range(1, len(disruption_status))}
    disrupt_trans.update({len(disruption_status) - 1: [len(disruption_status) - 1]})
    # shop_a = {'inside_room':{2:(0.95,0.85,0.5),3:()]}

    ## Model transitions
    shop_a = {0: [(1, 0.90), (2, 0.95), (2, 0.90), (2, 0.05), (0, 1.00), (5, 0.00), (0, 1.00), (7, 0.00)],  # nominal
              1: [(1, 0.90), (1, 0.95), (2, 0.50), (2, 0.15), (0, 1.00), (5, 0.00), (0, 1.00), (7, 0.00)],  # iceA
              2: [(0, 0.90), (0, 0.95), (2, 0.90), (2, 0.05), (0, 1.00), (5, 0.00), (0, 1.00), (7, 0.00)],  # iceB
              3: [(6, 0.90), (2, 0.95), (2, 0.90), (2, 0.05), (0, 1.00), (5, 0.00), (6, 0.90), (7, 0.00)],  # iceC
              4: [(1, 0.90), (1, 1.00), (2, 0.00), (2, 0.00), (0, 1.00), (5, 0.00), (0, 1.00), (7, 0.00)],  # alarmA
              5: [(1, 0.90), (2, 0.95), (2, 0.50), (2, 0.15), (0, 1.00), (5, 0.00), (0, 1.00), (7, 0.00)],  # alarmB
              6: [(1, 0.90), (1, 1.00), (2, 0.00), (2, 0.15), (0, 1.00), (5, 0.00), (0, 1.00), (7, 0.00)],  # alarmG
              }

    shop_b = {0: [(4, 0.90), (0, 0.90), (1, 0.10), (2, 0.05), (5, 0.90), (5, 0.90), (0, 1.00), (7, 0.00)],  # nominal
              1: [(1, 0.75), (1, 0.90), (1, 0.15), (2, 0.05), (0, 0.90), (5, 0.90), (0, 1.00), (7, 0.00)],  # iceA
              2: [(0, 0.90), (0, 0.95), (1, 0.10), (2, 0.05), (0, 0.90), (4, 0.50), (0, 1.00), (7, 0.00)],  # iceB
              3: [(6, 0.90), (0, 0.90), (1, 0.10), (2, 0.05), (0, 0.90), (5, 0.90), (6, 0.90), (7, 0.00)],  # iceC
              4: [(4, 0.90), (0, 1.00), (1, 0.00), (2, 0.00), (5, 0.90), (5, 0.90), (0, 1.00), (7, 0.00)],  # alarmA
              5: [(4, 0.90), (0, 0.90), (1, 0.10), (2, 0.05), (0, 1.00), (5, 0.05), (0, 1.00), (7, 0.00)],  # alarmB
              6: [(4, 0.90), (0, 1.00), (1, 0.00), (2, 0.00), (0, 1.00), (5, 0.05), (0, 1.00), (7, 0.00)],  # alarmG
              }

    repair = {0: [(0, 0.00), (1, 0.00), (1, 0.25), (2, 0.35), (5, 0.50), (5, 0.35), (0, 1.00), (7, 0.00)],  # nominal
              1: [(1, 0.75), (1, 0.80), (1, 0.05), (2, 0.05), (0, 0.80), (5, 0.35), (0, 1.00), (7, 0.00)],  # iceA
              2: [(0, 0.80), (0, 0.75), (1, 0.05), (2, 0.35), (0, 0.80), (5, 0.15), (0, 1.00), (7, 0.00)],  # iceB
              3: [(6, 0.75), (1, 0.00), (1, 0.25), (2, 0.35), (0, 0.80), (5, 0.35), (6, 0.80), (7, 0.00)],  # iceC
              4: [(0, 0.00), (0, 1.00), (1, 0.05), (2, 0.95), (5, 0.50), (5, 0.35), (0, 1.00), (7, 0.00)],  # alarmA
              5: [(0, 0.00), (1, 0.00), (1, 0.25), (2, 0.35), (0, 0.95), (5, 0.05), (0, 1.00), (7, 0.00)],  # alarmB
              6: [(0, 0.00), (0, 1.00), (1, 0.05), (2, 0.95), (0, 0.95), (5, 0.05), (0, 1.00), (7, 0.00)],  # alarmG
              }

    shopper = {0: [(0, 0.00), (1, 0.00), (1, 0.65), (2, 0.05), (5, 0.50), (5, 0.65), (0, 0.75), (7, 0.05)],  # nominal
               1: [(1, 0.90), (1, 0.90), (1, 0.35), (2, 0.05), (0, 0.75), (5, 0.65), (0, 0.75), (7, 0.05)],  # iceA
               2: [(0, 0.90), (0, 0.90), (1, 0.65), (2, 0.05), (0, 0.75), (5, 0.35), (0, 0.90), (7, 0.05)],  # iceB
               3: [(6, 0.90), (0, 0.90), (1, 0.65), (2, 0.05), (0, 0.75), (5, 0.65), (6, 0.90), (7, 0.00)],  # iceC
               4: [(0, 0.00), (2, 0.00), (2, 0.05), (3, 0.95), (5, 0.50), (5, 0.65), (0, 0.75), (7, 0.05)],  # alarmA
               5: [(0, 0.00), (1, 0.00), (1, 0.65), (2, 0.05), (0, 1.00), (5, 0.05), (0, 0.75), (7, 0.05)],  # alarmB
               6: [(0, 0.00), (2, 0.00), (2, 0.05), (3, 0.95), (0, 1.00), (5, 0.05), (0, 0.75), (7, 0.05)],  # alarmG
               }

    threat = {0: [(0, 0.00), (1, 0.00), (1, 0.65), (2, 0.05), (5, 0.50), (5, 0.65), (0, 0.75), (7, 0.05)],  # nominal
              1: [(1, 0.90), (2, 0.65), (1, 0.25), (2, 0.30), (0, 0.75), (5, 0.65), (0, 0.75), (7, 0.05)],  # iceA
              2: [(1, 0.05), (2, 0.50), (1, 0.25), (2, 0.30), (0, 0.75), (5, 0.35), (0, 0.90), (7, 0.05)],  # iceB
              3: [(1, 0.90), (2, 0.65), (1, 0.65), (2, 0.05), (0, 0.75), (5, 0.65), (0, 0.50), (7, 0.05)],  # iceC
              4: [(0, 0.00), (2, 0.50), (2, 0.05), (3, 1.00), (5, 0.50), (5, 0.65), (0, 0.75), (7, 0.05)],  # alarmA
              5: [(0, 0.00), (1, 0.00), (1, 0.65), (2, 0.05), (0, 1.00), (5, 0.00), (0, 0.75), (7, 0.05)],  # alarmB
              6: [(0, 0.00), (2, 0.50), (2, 0.05), (3, 1.00), (0, 1.00), (5, 0.00), (0, 0.75), (7, 0.05)],  # alarmG
              }

    home = {0: [(0, 0.00), (1, 0.00), (1, 0.15), (2, 0.05), (0, 0.75), (5, 0.15), (7, 0.85), (7, 0.75)],  # nominal
            1: [(1, 0.90), (1, 0.90), (1, 0.05), (2, 0.00), (0, 0.90), (5, 0.15), (0, 0.90), (7, 0.75)],  # iceA
            2: [(0, 0.90), (0, 0.90), (1, 0.05), (2, 0.00), (0, 0.90), (5, 0.05), (0, 0.90), (7, 0.25)],  # iceB
            3: [(6, 0.90), (0, 0.90), (1, 0.15), (2, 0.05), (0, 0.90), (5, 0.15), (6, 0.90), (7, 0.25)],  # iceC
            4: [(0, 0.00), (2, 0.00), (1, 0.05), (2, 0.95), (0, 0.75), (5, 0.15), (7, 0.85), (7, 0.75)],  # alarmA
            5: [(0, 0.00), (1, 0.00), (1, 0.15), (2, 0.05), (0, 1.00), (5, 0.05), (7, 0.85), (7, 0.75)],  # alarmB
            6: [(0, 0.00), (2, 0.00), (1, 0.05), (2, 0.95), (0, 1.00), (5, 0.05), (7, 0.85), (7, 0.75)],  # alarmG
            }

    trans_lists = [shop_a, shop_b, repair, shopper, threat, home]

    # nominal
    list_trans = [[] for t_s in trans_lists]

    for idx, t_list in enumerate(trans_lists):
        for a_i in t_list:
            for e_t in env_trans:
                for e_s in env_trans[e_t]:
                    sub_trans = methods[e_t](e_s, t_list[a_i][e_s][0], copy.deepcopy(env_trans[e_t][e_s]), a_i,
                                             t_list[a_i][e_s][1])
                    for s_b in sub_trans:
                        for d_s in disrupt_trans:
                            for d_t in disrupt_trans[d_s]:
                                if len(disrupt_trans[d_s]) == 1:
                                    list_trans[idx] += [(product_reverse[(s_b[0], d_s)], s_b[1],
                                                         product_reverse[(s_b[2], d_t)],
                                                         perturbed_prob(s_b[3], deltas[d_s], 1))]
                                else:
                                    if d_s == d_t:
                                        list_trans[idx] += [(product_reverse[(s_b[0], d_s)], s_b[1],
                                                             product_reverse[(s_b[2], d_t)],
                                                             perturbed_prob(s_b[3], deltas[d_s],
                                                                            1 - escalation_prob[a_i]))]
                                    else:
                                        list_trans[idx] += [(product_reverse[(s_b[0], d_s)], s_b[1],
                                                             product_reverse[(s_b[2], d_t)],
                                                             perturbed_prob(s_b[3], deltas[d_s], escalation_prob[a_i]))]
    mdp_lists = [MDP(states=list(product_states.keys()), alphabet=event_space, transitions=l_t) for l_t in list_trans]
    return mdp_lists, product_states, product_reverse


def track_outs(trans_in):
    # Paths in gridworld for transitions
    env_tracks = {(0, 1): [555, 554, 553, 552, 551, 550, 549, 548, 518],
                  (0, 0): [555, 554, 553, 583, 584, 585, 586, 556, 555],
                  (0, 6): [555, 556, 557, 558, 559, 560, 561, 562, 532],
                  (0, 4): [555, 585, 615, 645, 675, 705, 735, 765, 765],
                  (1, 0): [518, 519, 520, 521, 522, 523, 524, 525, 555],
                  (1, 1): [518, 517, 516, 546, 547, 548, 549, 548, 518],
                  (1, 2): [518, 488, 458, 428, 429, 399, 398, 397, 427],
                  (2, 1): [428, 429, 459, 458, 428, 458, 488, 518, 518],
                  (2, 2): [428, 429, 399, 369, 368, 367, 397, 427, 428],
                  (2, 3): [428, 398, 368, 338, 339, 340, 341, 342, 343],
                  (3, 2): [313, 312, 311, 310, 309, 308, 338, 368, 398],
                  (3, 3): [313, 283, 253, 223, 224, 254, 284, 314, 313],
                  (4, 0): [765, 766, 736, 706, 676, 646, 616, 586, 556],
                  (4, 5): [765, 765, 795, 825, 824, 854, 884, 885, 855],
                  (5, 4): [855, 856, 826, 825, 795, 796, 766, 765, 765],
                  (5, 5): [855, 856, 857, 887, 886, 885, 884, 854, 855],
                  (6, 0): [532, 531, 530, 529, 528, 527, 526, 525, 555],
                  (6, 6): [532, 533, 534, 564, 563, 562, 562, 562, 532],
                  (6, 7): [532, 502, 472, 473, 443, 413, 412, 442, 442],
                  (7, 6): [442, 441, 440, 470, 471, 472, 502, 532, 532],
                  (7, 7): [412, 382, 381, 380, 410, 440, 470, 471, 472]}
    return env_tracks[trans_in]
