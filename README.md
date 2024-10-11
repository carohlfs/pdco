
# Code used to generate the results for "Optimal Bandwidth Selection for Kernel Regression Using a Fast Grid Search and a GPU," published in PDCO 2017

This repository contains the programs used to produce the results in the paper "Optimal Bandwidth Selection for Kernel Regression Using a Fast Grid Search and a GPU," by Chris Rohlfs and Mohamed Zahran.

## Table of Contents
- [Overview](#overview)
- [Programs](#programs)
- [Compilation](#compilation)
- [Running the Code](#running-the-code)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [Citing This Work](#citing-this-work)
- [Authors](#authors)
- [License](#license)

## Overview

The following programs are included in this repository:

- `crossv_sequential.c`: A C program that performs a sorting-based grid search algorithm for optimal bandwidth selection without parallelism.
- `crossv.cu`: A CUDA program that performs the same operation as `crossv_sequential.c` but utilizes GPU for faster execution.
- `bandwidth_grid.R`: An R script for nonparametric regression, including multicore parallel cross-validation bandwidth selection using the `multicore` package.
- `call.script.R`: An R script that generates random data and compares runtime performance of custom bandwidth estimation and the function from the `np` R package.

## Programs

### crossv_sequential.c
C code that identifies the optimal bandwidth using a sorting-based grid search algorithm. This program does not take advantage of parallelism.

### crossv.cu
CUDA code that performs the same operation as `crossv_sequential.c`, but uses GPU parallelism to reduce runtime. This program is designed to be compatible with older GPUs by using low compute capability.

### bandwidth_grid.R
This R program performs nonparametric regression tasks, including bandwidth selection using multicore parallelism. It is called by `call.script.R` but can also be run independently.

### call.script.R
Generates random data and checks the runtimes of both custom bandwidth estimation code and the bandwidth estimation function in the `np` package.

## Compilation

To compile the C and CUDA programs, use the following commands:

```bash
# Compile crossv_sequential.c
gcc crossv_sequential.c -o crossv_sequential

# Compile crossv.cu
nvcc crossv.cu -o crossv
```

The CUDA program is compiled with a low compute capability to ensure compatibility with older GPUs.

## Running the Code

Both programs accept optional command-line arguments:

1. Sample size (default: 1024)
2. Number of bandwidths to consider (default: 50)
3. The length of the range of bandwidths (default: max X - min X)
4. The minimum bandwidth value (default: range / number of bandwidths)

Example usage:

```bash
./crossv_sequential 1024 50 1.0 0.02
./crossv 1024 50 1.0 0.02
```

## Dependencies

For the R scripts, you may need to install the `np` package. Run the following inside R:

```r
install.packages('np')
```

Other necessary packages:

- `multicore` (for parallel processing in R)

## How to Use

1. Run the R script from the command line:
   ```bash
   Rscript call.script.R
   ```
2. If you need to interactively run the script inside R:
   - Open R by typing `R` at the command prompt.
   - Paste each line from `call.script.R` into the R command line.

## Citing This Work

If you use this code or the results from the paper in your work, please cite the following publication:

```bibtex
@inproceedings{rohlfs2017optimal,
  title={Optimal Bandwidth Selection for Kernel Regression Using a Fast Grid Search and a GPU},
  author={Chris Rohlfs and Mohamed Zahran},
  booktitle={7th IEEE Workshop Parallel / Distributed Computing and Optimization (PDCO 2017), in conjunction with 31st IEEE International Parallel \& Distributed Processing Symposium (IPDPS)},
  year={2017},
  doi={10.1109/ipdpsw.2017.130},
  url={https://www.mzahran.com/pdco-13.pdf}
}
```

## Authors

- **Chris Rohlfs** - [GitHub](https://github.com/carohlfs)
- **Mohamed Zahran** - [Personal Website](https://www.mzahran.com)

## License

This project is licensed under the [MIT License](LICENSE).