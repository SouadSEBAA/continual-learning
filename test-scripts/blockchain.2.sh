#!/bin/bash

python3 main.py --bc --bc-iid -bc-nd 10 -bc-max_ncomm 10 -bc-ha 6,2,2 -bc-aio 1 -bc-pow 0 -bc-ko 3 -bc-nm 0 -bc-vh 0.20 -bc-cs 0 --experiment 5GNIDD --scenario class --iters 100 --fc-layers 4 --fc-units 300 --contexts 8 --replay buffer --structure 2 --batch 32
