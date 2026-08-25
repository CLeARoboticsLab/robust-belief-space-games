#!/bin/bash
cd ~/robust-belief-space-games
~/.juliaup/bin/julia --project=. ~/run_lowc_bigd.jl > ~/lowc_bigd.log 2>&1
