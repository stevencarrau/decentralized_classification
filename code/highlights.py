from live_sim import *

if not (os.path.exists(agents_save_path) and os.path.isdir(agents_save_path)):
    raise Exception("live_sim.py was probably not run yet. Run it to generate data and make sure"
                    f" the data is saved correctly. Should be stored at {agents_full_fpath}")

def main():
    """
    code to show highlights for a specific agent(s ?).
    relies on `live_sim.py` to be previously run and data to be previously
    saved. Loads that data and runs simulations again while preloading that
    data.
    """

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
            continue

        prev_state = int(chosen_agent.highlight_reel.get_item_value(i, "prev_state"))
        next_state = int(chosen_agent.highlight_reel.get_item_value(i, "next_state"))
        # print(f"from {prev_state} to {next_state}:")

        # takes the form of (0, [345, 512, ...])
        track = track_outs((prev_state, next_state))
        # print(track_queue)

        # form agent_indices just for the 1 agent
        highlight_agent_indices = [(chosen_agent_idx, prev_state)]
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

        # store the time step
        time_step = chosen_agent.highlight_reel.get_item_value(i, "time_step")

        # reset animation stuff so things run
        SimulationRunner.instance = None
        anim = None

        # run the simulation but with more parameters to indicate we want to run highlights
        _, anim = run_simulation(agent_indices=highlight_agent_indices, event_names=event_mapping,
                                 highlight_agent_idx=chosen_agent_idx, preloaded_track=track,
                                 preloaded_triggers=triggers, preloaded_prev_beliefs=prev_beliefs,
                                 preloaded_time_step=time_step, preloaded_delta_beliefs=delta_beliefs)

    print("Finished saving all highlights.")

if __name__ == "__main__":
    main()