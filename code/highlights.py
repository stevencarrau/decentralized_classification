from live_sim import *
from darpa_model import get_env_tracks
import os
import time

if not (os.path.exists(agents_save_path) and os.path.isdir(agents_save_path)):
    raise Exception("live_sim.py was probably not run yet. Run it to generate data and make sure"
                    f" the data is saved correctly. Should be stored at {agents_full_fpath}")

def main():
    """
    code to show highlights for a specific agent.
    relies on `live_sim.py` to be previously run and data to be previously
    saved. Loads that data and runs simulations again while preloading that
    data.
    usage: `python highlights.py AGENT_IDX`
    """

    # create highlights folder if does not exist
    highlight_videos_save_path = "highlight_videos"
    Util.prepare_dir(highlight_videos_save_path)

    agents = np.load(agents_full_fpath, allow_pickle=True).tolist()
    # print(agents)

    # get the agent index from args
    assert len(sys.argv) == 2, "Must specify the agent idx through program arguments."
    chosen_agent_idx = int(sys.argv[1])
    agent_idx_choices = [agent.agent_idx for agent in agents]
    if chosen_agent_idx not in agent_idx_choices:
        raise Exception(f"Invalid agent idx: must be one of {agent_idx_choices} for the saved "
                        f"agents data file {agents_full_fpath}")

    chosen_agent = get_agent_with_idx(chosen_agent_idx, agents)

    # load the most significant highlight data for the agent
    highlights = chosen_agent.highlight_reel.reel
    # for every highlight, run an animation. The most important highlights are last
    # due to the backend sorting in the highlight reel, so start from the back (most
    # important episodes first)
    anim = None
    for i in range(len(highlights) - 1, -1, -1):
        print(f"running highlight {i}: {chosen_agent.highlight_reel.reelitem2dict(i)}")
        if np.array_equal(highlights[i], Agent.HighlightReel.EMPTY_ITEM):
            return

        prev_state = chosen_agent.highlight_reel.get_item_value(i, "prev_state")
        next_state = chosen_agent.highlight_reel.get_item_value(i, "next_state")
        # print(f"from {prev_state} to {next_state}:")

        # get the track for the agent to replay based off of stored state data
        transformed_prev_state = Util.prod2state(prev_state, chosen_agent.states)
        transformed_next_state = Util.prod2state(next_state, chosen_agent.states)
        track = track_outs((transformed_prev_state, transformed_next_state))
        # print(track_queue)

        # form agent_indices just for the 1 agent
        highlight_agent_indices = [(chosen_agent_idx, transformed_prev_state)]
        # print(highlight_agent_indices)

        # get trigger
        triggers = []
        trigger = int(chosen_agent.highlight_reel.get_item_value(i, "trigger"))
        # a trigger of 6 means images from events 4 and 5 since it triggers both alarms:
        # have both present in the triggers list.
        # also, don't add to triggers if it is trigger 0 since that is just the nominal effect
        # (nothing is shown)
        if trigger == 6:
            triggers.extend([4, 5])
        elif trigger != 0:
            triggers.append(trigger)

        # get belief values
        # values that the agent held in prev_state
        prev_beliefs = chosen_agent.highlight_reel.get_item_value(i, "prev_beliefs")
        # delta from prev_beliefs to the new beliefs
        delta_beliefs = chosen_agent.highlight_reel.get_item_value(i, "delta_beliefs")

        # get the time step
        time_step = chosen_agent.highlight_reel.get_item_value(i, "time_step")

        # get the most likely alternate state that the agent would have transitioned to alternatively:
        all_next_alt_states = chosen_agent.mdp.next_states_sorted_prob(s=prev_state, a=trigger)
        alt_track = None
        for alt_state in all_next_alt_states:
            # get first track which produces a valid track since some
            # (prev_state, alt_state) tuples may not be in the gridworld transition model
            trans_in = (transformed_prev_state, Util.prod2state(alt_state, chosen_agent.states))
            if trans_in in get_env_tracks():
                # print("alt_state found:", alt_state)
                alt_track = track_outs(trans_in)
                # we're trying to show an alternate track than the one
                # chosen, so skip if we encounter the same track that the agent actually
                # took
                if alt_track == track:
                    continue
                else:
                    break

        assert alt_track is not None, "Some track should have been found"
        alternate_tracks = [alt_track]

        # reset animation stuff so things run
        SimulationRunner.instance = None
        anim = None
        # remove previous frames
        import shutil
        shutil.rmtree(video_data_save_path)
        # create the dir again
        Util.prepare_dir(video_data_save_path)

        # run the simulation but with more parameters to indicate we want to run highlights
        _, anim = run_simulation(agent_indices=highlight_agent_indices, event_names=event_mapping,
                                 highlight_agent_idx=chosen_agent_idx, preloaded_track=track,
                                 preloaded_triggers=triggers, preloaded_prev_beliefs=prev_beliefs,
                                 preloaded_time_step=time_step, preloaded_delta_beliefs=delta_beliefs,
                                 preloaded_alternate_tracks=alternate_tracks)

        agent_character_names = ['Captain_America', 'Black_Widow', 'Hulk', 'Thor', 'Thanos', 'Ironman']
        agent_name = agent_character_names[chosen_agent_idx]
        vid_title = f"{agent_name}_Highlight_Timestep_{time_step}.mp4"
        full_video_path = f"{highlight_videos_save_path}/{vid_title}"

        print(f"Saving highlight for time step {time_step} at \"{full_video_path}\" ...")
        # use ffmpeg to patch the videos together
        os.system(f"ffmpeg -r 0.80 -i {video_data_save_path}/%04d.png -vcodec mpeg4 -y {full_video_path}")
        print(f"Finished saving highlight for time step {time_step}.")


    print("Finished saving all highlights.")

if __name__ == "__main__":
    main()