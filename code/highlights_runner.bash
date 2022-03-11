#!/usr/bin/bash

agent_idx=1
for highlight_reel_idx in {0..4}
  do
    python highlights.py $agent_idx $highlight_reel_idx
  done

echo "Finished saving all highlights."