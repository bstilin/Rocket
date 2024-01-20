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

## Note on Multi-Threading
 `Rocket.jl` and `Rocket_mv.jl` both use the Threads.@threads macro to speed up computation. See the Multi-Threading [documentation](https://docs.julialang.org/en/v1/manual/multi-threading/). The Julia discourse is useful for finding information on setting the number of available threads. 

## Contents

### source

- `Rocket.jl`: A Julia implementation of the ROCKET transform as introduced in [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://link.springer.com/article/10.1007/s10618-020-00701-z). Note that this code only implements the ROCKET transform. As such, it does NOT perform standardization of the input data. Further, it assumes that the time series samples are all of the same length and do not have any missing data.

- `RocketHelperFunctions.jl`: A collection of functions useful for interfacing the ROCKET transform with classification algorithms.

- `RocketHelperFunctions_mv.jl`: A collection of functions useful for interfacing the multivariate ROCKET transform with classification algorithms.

- `Rocket_mv.jl`: A Julia implementation of the ROCKET transform as introduced in [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://link.springer.com/article/10.1007/s10618-020-00701-z) adapted for multivariable time series. The original paper does not prescribe a specific way of adapting ROCKET to multivariable time series. For details on the adaptation method used here, see below and docstring. This code does NOT perform standardization of the input data. Further, it assumes that the time series samples are all of the same length and do not have any missing data.

### data

A folder in which to store your data. The examples assume the UCR data is contained in this folder.

### examples
Example datasets were taken from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/)

- `CoffeeExample.jl`: A univariate, binary classification problem using the ridge regression implementation from MLJLinearModels.jl. The code assumes your data folder contains the folder "Coffee" from the UCR archive.

- `ChlorineConcentrationExample.jl`: A univariate, three-class classification problem using the ridge regression implementation from ScikitLearn.jl. See Python documentation for `sklearn.linear_model.RidgeClassifier` for details of how ridge regression is applied to a multiclass problem. The code assumes your data folder contains the folder "ChlorineConcentration" from the UCR archive.
  
- `ClassifierExamples.jl`: A collection of examples demonstrating how to run L1 or elastic-net penalized logistic regression, LASSO, and ridge regression using MLJLinearModels.jl and ScikitLearn.jl.

- `rocket_mv.ipynb`: A demonstration of rocket applied to multivariate time series classification, with one multiclass example with 9 channels, ([`ArticularyWordRecognition`](http://www.timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition)), one multiclass example with 2 channels ([`PenDigits`](http://www.timeseriesclassification.com/description.php?Dataset=PenDigits)), and one with three classes and two channels ([`AtrialFibrillation`](http://www.timeseriesclassification.com/description.php?Dataset=AtrialFibrillation)), and a demonstration of a hybrid approach using ROCKET and catch22 ([`catch22: CAnonical Time-series CHaracteristics`](https://link.springer.com/article/10.1007/s10618-019-00647-x)) as a proof of concept to combine relevant time series features for classification.

