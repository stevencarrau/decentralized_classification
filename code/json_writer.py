import json


def write_JSON(filename, data):
    with open(filename, 'w') as outfile:
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


def all_agent_tracks(list_agent_names, list_tracks):
    dict_out = dict()
    tot_t = len(list_tracks[0])
    for i in range(tot_t):
        time_dict = dict()
        for a_i, t_i in zip(list_agent_names, list_tracks):
            agent_dict = dict({'AgentLoc': t_i[i]})
            if i == 0:
                agent_dict.update({'Id_no': list_agent_names})
            time_dict.update({a_i: agent_dict})
        dict_out.update({i: time_dict})

    return dict_out


# agents = [0, 1, 2, 3, 4, 5]
# # store A owner (Andy), store B owner (Barney), customer C (Chloe), customer D (Dora), customer E (Edward), robot
# tracks = [[397, 398, 399, 429, 428, 427, 457, 458, 459, 460], [854, 824, 825, 826, 827, 857, 856, 855, 854, 853],
#           [543, 546, 549, 489, 459, 460, 458, 489, 549, 552], [573, 576, 578, 488, 428, 368, 338, 278, 248, 250],
#           [723, 726, 729, 732, 735, 765, 795, 825, 826, 856], [633, 636, 639, 642, 645, 648, 651, 654, 657, 660]]
# agent_paths = all_agent_tracks(agents, tracks)
# write_JSON('AgentPaths_pink_bad.json', agent_paths)
#
# with open('AgentPaths_pink_bad.json') as json_file:
#     data = json.load(json_file)

# print(data['1']['3']['AgentLoc'])
