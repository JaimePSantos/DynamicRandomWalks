# Discrete Time Random Walk Simulator

This project aims to firstly investigate discrete-time random walks in dynamic graphs, by studying properties such as the cover time. The second step will be to apply these findings to quantum walks.

The code might later be converted to a python package, so the following sections are a proof of concept, although you will need the dependencies if you want to experiment with the notebooks.

## Introduction

`DiscreteTimeRandomWalker` is a Python package for simulating and analyzing discrete-time random walks on various structures such as grids, trees, and networks. Through intuitive APIs and functions, this simulator aims to aid researchers, students, and enthusiasts in understanding the properties and behavior of discrete time random walks.

## Features

- Simulate random walks on different structures such as 1D lines, 2D grids, trees, and arbitrary networks.
- Visualize the random walk trajectories and probability distributions.
- Analyze hitting times, return times, and cover times.
- Leverage `qwak-sim` for quantum walk simulations.
- Employ symbolic mathematics to analyze random walks with SymPy.

## Dependencies

- Numpy
- Scipy
- SymPy
- Matplotlib
- NetworkX
- qwak-sim

> **Note**: `qwak-sim` is a separate package for simulating quantum walks. This is mainly used for the plotting functions.

## Installation

### Prerequisites

Ensure that you have Python 3.7 or higher installed on your system. You can download Python [here](https://www.python.org/downloads/).

### Installing DiscreteTimeRandomWalker

1. Clone the repository to your local machine.
    ```bash
    git clone https://github.com/YourUsername/DTRandomWalk.git
    ```
2. Navigate to the cloned repository.
    ```bash
    cd DTRandomWalk
    ```
3. Install the required dependencies.
    ```bash
    pip install numpy scipy sympy matplotlib networkx qwak-sim
    ```

## Documentation & Contribution

Work in progress.