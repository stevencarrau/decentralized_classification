import json


def write_JSON(filename,data):
    with open(filename,'w') as outfile:
        json.dump(stringify_keys(data), outfile)

def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise

            # delete old key
            del d[key]
    return d



def all_agent_tracks(list_agent_names,list_tracks):
	dict_out = dict()
	tot_t = len(list_tracks[0])
	for i in range(tot_t):
		time_dict = dict()
		for a_i,t_i in zip(list_agent_names,tracks):
			agent_dict = dict({'AgentLoc':t_i[i]})
			if i==0:
				agent_dict.update({'Id_no':list_agent_names})
			time_dict.update({a_i:agent_dict})
		dict_out.update({i:time_dict})

	return dict_out

agents = [0,1,2,3]
tracks = [[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19],[20,21,22,23,24,25,26,27,28,29],[30,31,32,33,34,35,36,37,38,39]]
agent_paths = all_agent_tracks(agents,tracks)
write_JSON('AgentPaths.json',agent_paths)

with open('AgentPaths.json') as json_file:
	data = json.load(json_file)

print(data['1']['3']['AgentLoc'])