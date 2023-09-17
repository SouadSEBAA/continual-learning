#!/bin/bash

source test-venv/bin/activate

python3 main.py --experiment 5GNIDD --scenario class --replay buffer --contexts 8  --structure 2 --fc-layers 4 --fc-units 300 --iters 5 --bc -bc-nd 9 -bc-ha 5,4,0 -bc-nm 2 -bc-max_ncomm 2 -bc-vh 1.0 -bc-ko 2 --bc-cpu --bc-iid -bc-aio 1 -bc-pow 0 -bc-cs 0 --bc-cons poa --batch 32
