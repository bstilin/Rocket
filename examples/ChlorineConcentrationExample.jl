import CSV
import Tables

using DataFrames
using CategoricalArrays
using ScikitLearn

include("Rocket.jl")

@sk_import linear_model: RidgeClassifierCV

#Load Data. Assumes data folder contains folder "ChlorineConcentration" from the UCR archive
dataset_name="ChlorineConcentration" 
file_path_test=joinpath("data",dataset_name,dataset_name*"_TEST.tsv")
file_path_train=joinpath("data",dataset_name,dataset_name*"_TRAIN.tsv")

test=CSV.File(file_path_test,delim="\t") |> Tables.matrix
train=CSV.File(file_path_train,delim="\t") |> Tables.matrix

X_train = train[:,2:end]; Y_train=train[:,1] #Classes are first column
X_test = test[:,2:end]; Y_test=test[:,1] #Classes are first column


#Generate kernels and apply ROCKET transform

num_kernels=10000
K = generate_kernels(size(test)[2],num_kernels);

X_train_transform,_,_ = Rocket_transform(X_train,K);
X_test_transform,_,_ = Rocket_transform(X_test,K);

#Fit
sk_model=RidgeClassifierCV()
sk_model.fit(X_train_transform,Y_train)

#Predict
Y_pred=sk_model.predict(X_test_transform)

#Calculate Accuracy
sum(Y_pred .== Y_test)/length(Y_pred) #ROCKET paper accuracy: .8130