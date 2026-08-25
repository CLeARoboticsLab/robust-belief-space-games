#!/bin/bash
cd ~/robust-belief-space-games
~/.juliaup/bin/julia --project=. ~/run_p2types.jl > ~/p2types.log 2>&1
