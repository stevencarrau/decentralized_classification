import json
import pandas as pd
import csv
from ast import literal_eval
import pickle
# fname = '5agents_3-Z_range_async_min'
# with open(fname+'.json') as json_file:
#     data3 = json.load(json_file)
#
# fname = '5agents_3-Z_range_sync_min'
# with open(fname+'.json') as json_file:
#     data = json.load(json_file)


# with open(fname+'.pickle','rb') as env_file:
# 	gwg = pickle.load(env_file)

# fname = '/home/scarr/Downloads/5agents_4-9_range_async_min'
# with open(fname+'.json') as json_file:
#     data = json.load(json_file)
#     data2 = data

fname = '5agents_6-HV_range_async_min'
with open(fname+'.json') as json_file:
    data2 = json.load(json_file)
data = data2
data3 = data



df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
frames = 75
categories = list(range(5))

def coords(s,ncols):
    return (int(s /ncols), int(s % ncols))


time = 8
id_keys = data2[str(time)].keys()
for id_no in id_keys:
    values = data2[str(time)][id_no]['ActBelief']
    cat_range = range(len(id_keys))
    value_dict = dict([[c_r, 0.0] for c_r in cat_range])
    for v_d in value_dict.keys():
        for k_i in values.keys():
            if literal_eval(k_i)[v_d] == 1:
                value_dict[v_d] += 1 * values[k_i]
    print('{} at {}: {}'.format(id_no,coords(data2[str(0)][id_no]['AgentLoc'][0],10),value_dict.values()))


# ---------- PART 2:

nrows = 10
ncols = 10
moveobstacles = []
obstacles = []


# #4 agents larger range
obs_range = 4
belief_good = df['0'][0]['GoodBelief']

agent_loc = dict()
local_belief = dict()
agent_belief_good_min = dict()
agent_belief_good_avg = dict()
agent_conn = dict()
for j in categories:
    agent_loc.update({j:dict()})
    agent_conn.update({j:dict()})
    local_belief.update({j:dict()})
    agent_belief_good_min.update({j:dict()})
    agent_belief_good_avg.update({j:dict()})

for i in range(frames):
    for id_no in categories:
        agent_loc[id_no].update({i:tuple(reversed(coords(df[str(i)][id_no]['AgentLoc'][0], ncols)))})
        local_belief[id_no].update({i:df3[str(i)][id_no]['LocalBelief'][belief_good]})
        agent_belief_good_min[id_no].update({i:df[str(i)][id_no]['ActBelief'][belief_good]})
        agent_belief_good_avg[id_no].update({i:df2[str(i)][id_no]['ActBelief'][belief_good]})

# with open('Local_Low_Var.csv', 'w') as f:
#     writer = csv.writer(f)
#     for time in agent_belief_good_min[0].keys():
#         row_tup = (time,)
#         for val in agent_loc.keys():
#             row_tup += (local_belief[val][time],agent_belief_good_min[val][time],agent_belief_good_avg[val][time],)
#         writer.writerow(row_tup)

with open('Local_U1_Var.csv', 'w') as f:
    writer = csv.writer(f)
    for time in agent_belief_good_min[0].keys():
        row_tup = (time,)
        for val in agent_loc.keys():
            row_tup += (local_belief[val][time],agent_belief_good_min[val][time],)
        writer.writerow(row_tup)