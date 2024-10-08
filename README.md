# Constraint Boundary Wandering Framework: Enhancing Constrained Optimization with Deep Neural Networks
This repository is by 
`Shuang Wu`,
[Shixiang Chen](https://chenshixiang.github.io/),
and [Leifei Zhang](https://dblp.org/pid/28/10770.html)
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our paper
"[Constraint Boundary Wandering Framework: Enhancing Constrained Optimization with Deep Neural Networks]"

If you find this repository helpful in your publications,
please consider citing our paper.


## Abstract

Constrained optimization problems are pervasive in various fields, and while conventional techniques offer solutions, they often struggle with scalability. Leveraging the power of
deep neural networks (DNNs) in optimization, we present a novel learning-based approach, the Constraint Boundary Wandering Framework (CBWF), to address these challenges. Our contri-
butions include introducing a Boundary Wandering Strategy(BWS) inspired by the active-set method, enhancing equality constraint feasibility, and treating the Lipschitz constant as a
learnable parameter. Additionally, we evaluate the regularization term, finding that the L2 norm yields superior results. Extensive testing on synthetic datasets and the ACOPT dataset demonstrates
CBWF’s superiority, outperforming existing deep learning-based solvers in terms of both objective and constraint loss.

## Dependencies

+ Python 3.x
+ [PyTorch](https://pytorch.org) >= 1.8
+ numpy/scipy/pandas
+ [osqp](https://osqp.org/): *State-of-the-art QP solver*
+ [qpth](https://github.com/locuslab/qpth): *Differentiable QP solver for PyTorch*
+ [ipopt](https://coin-or.github.io/Ipopt/): *Interior point solver*
+ [pypower](https://pypi.org/project/PYPOWER/): *Power flow and optimal power flow solvers*
+ [argparse](https://docs.python.org/3/library/argparse.html): *Input argument parsing*
+ [pickle](https://docs.python.org/3/library/pickle.html): *Object serialization*
+ [hashlib](https://docs.python.org/3/library/hashlib.html): *Hash functions (used to generate folder names)*
+ [setproctitle](https://pypi.org/project/setproctitle/): *Set process titles*
+ [waitGPU](https://github.com/riceric22/waitGPU) (optional): *Intelligently set `CUDA_VISIBLE_DEVICES`*



## Instructions

### Dataset generation

Datasets for the experiments presented in our paper are available in the `datasets` folder. These datasets can be generated by running the Python script `make_dataset.py` within each subfolder (`simple`, `nonconvex`, and `acopf`) corresponding to the different problem types we test. And we use the `make_dataset_high_ioopt.py` to generate the high-order objective case.

### Running experiments

Our method and baselines can be run using the following Python files:
+ `bws_main.py`: Our modified DC3 method 
+ `cbwf_method.py`: Our proposed method
+ `bws_second_main.py`: After training DC3, we use BWS to find the better solution



See each file for relevant flags to set the problem type and method parameters. Notably:
+ `--probType`: Problem setting to test (`cbwf_method` provides `simple`, `nonconvex`, `acopf57` or `high_o` and `bws_main` only surport `simple`, `nonconvex` and `acopf57`)
+ `--simpleVar`, `--simpleIneq`, `simpleEq`, `simpleEx`: If the problem setting is `simple`, the number of decision variables, inequalities, equalities, and datapoints, respectively.
+ `--nonconvexVar`, `--nonconvexIneq`, `nonconvexEq`, `nonconvexEx`: If the problem setting is `nonconvex`, the number of decision variables, inequalities, equalities, and datapoints, respectively.
+ Or you can altert this hyperparameter in  `default_args.py`


