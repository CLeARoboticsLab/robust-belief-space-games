#!/bin/bash
cd ~/robust-belief-space-games
~/.juliaup/bin/julia --project=. ~/run_p2types_morec.jl > ~/p2types_morec.log 2>&1
