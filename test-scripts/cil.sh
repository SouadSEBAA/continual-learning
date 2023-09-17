#!/bin/bash

source test-venv/bin/activate

python3 main.py --experiment 5GNIDD --scenario class --replay buffer --contexts 8  --structure 2 --fc-layers 4 --fc-units 300 --iters 5 
