#!/bin/bash
cd ~/robust-belief-space-games
~/.juliaup/bin/julia --project=. ~/run_p2types_fill.jl > ~/p2types_fill.log 2>&1
