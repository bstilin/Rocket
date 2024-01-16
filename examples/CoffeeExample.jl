import CSV
import Tables

using DataFrames
using MLJ
using CategoricalArrays

include("Rocket.jl")

@load RidgeRegressor pkg=MLJLinearModels

#Load Data. Assumes data folder contains folder "Coffee" from the UCR archive
dataset_name="Coffee"
file_path_test=joinpath("data",dataset_name,dataset_name*"_TEST.tsv")
file_path_train=joinpath("data",dataset_name,dataset_name*"_TRAIN.tsv")

test=CSV.File(file_path_test,delim="\t") |> Tables.matrix
train=CSV.File(file_path_train,delim="\t") |> Tables.matrix

X_train = train[:,2:end]; Y_train=train[:,1]; #Classes are first column
X_test = test[:,2:end]; Y_test=test[:,1]; #Classes are first column

Y_train[Y_train .== 0] .= -1; #Convert data from 0/1 to -1.0/1.0 for use with Ridge Regression
Y_test[Y_test .== 0] .= -1; #Convert data from 0/1 to -1.0/1.0 for use with Ridge Regression

#Generate kernels and apply ROCKET transform

num_kernels=1000
K = generate_kernels(size(test)[2],num_kernels);

X_train_transform,_,_ = Rocket_transform(X_train,K);
X_test_transform,_,_ = Rocket_transform(X_test,K);

X_train_transform = DataFrame(X_train_transform,:auto); #DataFrames are MLJ compatible
X_test_transform = DataFrame(X_test_transform,:auto); #DataFrames are MLJ compatible

#Create Model and Fit

model =MLJLinearModels.RidgeRegressor()

l,u = 0, 1 #range of lambdas to search in hyperparameter search
num_models=5  #number of models to fit in hyperparameter search
num_folds=5 #number of elements in partition of training data for cross validation

lambda_range = range(model, :lambda, lower=l, upper=u) #Defining range object for hyperparameter tuning

self_tuning_model= TunedModel(model=model,
                              resampling = CV(nfolds=num_folds, rng=1234),
                              tuning = RandomSearch(), #change to Grid() for random search
                              range = lambda_range,
                              n=num_models);

mach = machine(self_tuning_model, X_train_transform,Y_train)

#Fit Model
fit!(mach)

#Calculate Accuracy
Y_pred=sign.(predict_mode(mach,X_test_transform)) #round to 
sum(Y_pred .== Y_test)/length(Y_pred) #ROCKET paper accuracy: 1.000