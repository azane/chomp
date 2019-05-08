# WIP

# chomp
An implementation of some methods described in the CHOMP paper: https://www.ri.cmu.edu/pub_files/2013/5/CHOMP_IJRR.pdf

# Installation Instructions
Running at least Python 3.6.3, run `pip install requirements.txt`
I *highly* recommend using a `virtualenv` for this.

# Running Instructions
To run the comparison between straight-line initialized CHOMP vs. RRT initialized CHOMP, run:
`PYTHONPATH=${PYTHONPATH}:PATH/TO/REPO/chomp python3 experiments/exp1.py --vis`
For example, I run: `PYTHONPATH=${PYTHONPATH}:~/GitRepo/chomp python3 experiments/exp1.py --vis`
Note that this will not show a visualization until the end of the optimization process.

To run the visualization of the optimization process for CHOMP, run:
`PYTHONPATH=${PYTHONPATH}:PATH/TO/REPO/chomp python3 vis_src/vis_scenario1.py`
While this does show the visualization during optimization, it hangs significantly, so I suggest waiting until optimization succeeds or fails to pan around.
A successful optimization will show the "spaceship" moving along the path from start to goal.

To run a visualization of the simple RRT, run:
`PYTHONPATH=${PYTHONPATH}:PATH/TO/REPO/chomp python3 vis_src/vis_rrt.py`
