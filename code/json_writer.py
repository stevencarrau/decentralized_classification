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


def all_agent_tracks(list_agent_names, list_tracks,event_track=None,belief_track=None):
    dict_out = dict()
    tot_t = len(list_tracks[0])
    for i in range(tot_t):
        time_dict = dict()
        for a_i, t_i in zip(list_agent_names, list_tracks):
            agent_dict = dict({'AgentLoc': t_i[i]})
            if i == 0:
                agent_dict.update({'Id_no': list_agent_names})
            time_dict.update({a_i: agent_dict})
            if event_track is not None:
                time_dict.update({'Event':event_track[i]})
            if belief_track is not None:
                time_dict.update({'Belief':belief_track[i]})
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

# agents = [0, 1, 2, 3, 4, 5]
# triggers = ["normal", "fire_alarm_test", "ice_cream_truck_test"]
#
# # store A owner
# # capt_america_resting_pt = 397
# capt_america_track_normal = [397, 398, 399, 429, 428, 427, 457, 458, 459, 460]
# # capt_america_track_ice_cream = [397, 398, 399, 429, 429, 429, 429, 429, 429, 429]
#
# # store B owner
# # black_widow_resting_pt = 854
# black_widow_track_normal = [854, 824, 825, 826, 827, 857, 856, 855, 854, 853]
# # black_widow_track_ice_cream = [854, 824, 825, 826, 826, 826, 826, 826, 826, 826]
#
# # customer C - good
# # hulk_resting_pt = 543
# hulk_track_normal = [543, 545, 548, 488, 458, 459, 458, 488, 548, 551]
# # hulk_track_ice_cream = [543, 545, 548, 488, 488, 488, 488, 488, 488, 488]
#
# # customer D - bad
# # thanos_resting_pt = 573
# thanos_track_normal = [573, 576, 578, 488, 428, 368, 338, 278, 248, 250]
# # thanos_track_ice_cream = thanos_track_normal  # bad guys will keep going after the electric box
#
# # customer E - Thor
# # thor_resting_pt = 723
# thor_track_normal = [723, 726, 729, 732, 735, 765, 795, 825, 826, 856]
# # thor_track_ice_cream = [723, 726, 729, 732, 732, 732, 732, 732, 732, 732]
#
#
# # Robot
# # ironman_resting_pt = 633
# ironman_track_normal = [633, 636, 639, 642, 645, 648, 651, 654, 657, 660]
# ironman_track_ice_cream = ironman_track_normal   # since we are ironman, there's no reason for us to change course
#
#
# normal_tracks = [capt_america_track_normal, black_widow_track_normal, hulk_track_normal,
#                  thanos_track_normal, thor_track_normal, ironman_track_normal]
#
# # ice_cream_truck_tracks = [capt_america_track_ice_cream, black_widow_track_ice_cream, hulk_track_ice_cream,
# #                           thanos_track_ice_cream, thor_track_ice_cream, ironman_track_ice_cream]
#
# # police_donut_tracks = [capt_america_track_police_donut, black_widow_track_police_donut, hulk_track_police_donut,
# #                  thanos_track_police_donut, thor_track_police_donut, ironman_police_donut]
# #
# # fire_alarm_tracks= [capt_america_track_fire_alarm, black_widow_track_fire_alarm, hulk_track_fire_alarm,
# #                  thanos_track_fire_alarm, thor_track_fire_alarm, ironman_fire_alarm]
# #
# # maintenance_crew_tracks = [capt_america_track_maintenance_crew, black_widow_track_maintenance_crew,
# #                            hulk_track_maintenance_crew, thanos_track_maintenance_crew,
# #                            thor_track_maintenance_crew, ironman_track_maintenance_crew]
#
# for trigger in triggers:
#     agent_paths = None
#     # if test == "ice_cream_truck_test":
#     #     agent_paths = all_agent_tracks(agents, ice_cream_truck_tracks)
#     # elif test == "fire_alarm_test":
#     #     agent_paths = all_agent_tracks(agents, fire_alarm_tracks)
#     # elif test == "police_donut_test":
#     #     agent_paths = all_agent_tracks(agents, police_donut_tracks)
#     # elif test == "maintenance_crew_test":
#     #     agent_paths = all_agent_tracks(agents, maintenance_crew_tracks)
#     if trigger == "normal" or agent_paths is None:
#         trigger = "normal"
#         agent_paths = all_agent_tracks(agents, normal_tracks)
#
#     file_name = 'AgentPaths_{trigger}.json'.format(trigger=trigger)
#     write_JSON(file_name, agent_paths)
#     # print(file_name)
#     print(trigger)
#     #
#     # with open(file_name) as json_file:
#     #     data = json.load(json_file)
#
# # print(data['1']['3']['AgentLoc'])
# print("Successfully wrote all tracks to JSON")

