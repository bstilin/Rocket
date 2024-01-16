# Implementation of ROCKET
## Goal

This repository is designed to support two projects:

1. Determining if the seismic background noise of forested areas is different from unforested areas.
2. Determining if cell signaling patterns can help explain cell decision-making in response to stress.

In both cases, the data are time series that do not contain patterns easily visible to a human. We plan to train a ROCKET-based classifier and then perform feature selection on the ROCKET kernels. We expect that the most significant ROCKET kernels for classification may contain information about relevant patterns in the data, ideally providing scientific insight.

Keeping in mind that others may be interested in this approach, we have kept the implementation of ROCKET lightweight and with easily interpretable code to allow ourselves and others to incorporate the domain knowledge unique to a problem of interest into the algorithm. Domain knowledge may inform:

1. Custom kernel pooling functions beyond the PPV and max pooling done in the original ROCKET paper.
2. Custom kernel generation.
3. The inclusion of meta-data or other time series features in the kernel space regression (see Catch-22 example).

Creating more explicit support for these features is planned for the future. Systematic benchmarking against the Scikit-learn implementation of ROCKET, beyond the several examples contained already, is also in the works as well as support for repeatable testing using StableRNGs.jl.

## Documentation

See the source code and use Julia's built-in help mode to access docstrings. The docstrings are written to be the main source of documentation.

## Contents

### source

- `Rocket.jl`: A Julia implementation of the ROCKET transform as introduced in [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](arXiv:1910.13051). Note that this code only implements the ROCKET transform. As such, it does NOT perform standardization of the input data. Further, it assumes that the time series samples are all of the same length and do not have any missing data.

- `RocketHelperFunctions.jl`: A collection of functions useful for interfacing the ROCKET transform with classification algorithms.

- `Rocket_mv.jl`: A Julia implementation of the ROCKET transform as introduced in [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](arXiv:1910.13051) adapted for multivariable time series. The original paper does not prescribe a specific way of adapting ROCKET to multivariable time series. For details on the adaptation method used here, see below and docstring. This code does NOT perform standardization of the input data. Further, it assumes that the time series samples are all of the same length and do not have any missing data.

- `ClassifierExamples.jl`: A collection of examples demonstrating how to run L1 or elastic-net penalized logistic regression, LASSO, and ridge regression using MLJLinearModels.jl and ScikitLearn.jl.

### data

A folder in which to store your data. The examples assume the UCR data is contained in this folder.

### examples
Example datasets were taken from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/)

- `CoffeeExample.jl`: A univariate, binary classification problem using the ridge regression implementation from MLJLinearModels.jl. The code assumes your data folder contains the folder "Coffee" from the UCR archive.

- `ChlorineConcentrationExample.jl`: A univariate, three-class classification problem using the ridge regression implementation from ScikitLearn.jl. See Python documentation for `sklearn.linear_model.RidgeClassifier` for details of how ridge regression is applied to a multiclass problem. The code assumes your data folder contains the folder "ChlorineConcentration" from the UCR archive.
