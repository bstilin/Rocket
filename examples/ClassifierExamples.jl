#Requires install of MLJLinearModels as well as below packages. MLJLinearModels docstrings are the best source of documentation for it

import ScikitLearn
import MLJ
using Random
using DataFrames
using Statistics
using CategoricalArrays

ScikitLearn.@sk_import linear_model: RidgeClassifierCV
ScikitLearn.@sk_import linear_model: LogisticRegressionCV
MLJ.@load LogisticClassifier pkg=MLJLinearModels
MLJ.@load RidgeRegressor pkg=MLJLinearModels
MLJ.@load LassoRegressor pkg=MLJLinearModels
MLJ.@load ElasticNetRegressor pkg=MLJLinearModels



#loading dataset
X, y = MLJ.@load_crabs #MLJ standard dataset

#Converting X to a dataframe
df=DataFrame(X)

#standardizing data
foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(df))

#Test/Train Split 
N=size(df)[1] #number of observations
split=.8 #percent used for training data

train_ind,test_ind=MLJ.partition(shuffle(1:N),split)

train_X=df[train_ind,:];test_X=df[test_ind,:]
train_y=y[train_ind]; test_y=y[test_ind]

#Run all of the above code and then move to the section about the classifier you wish to work with. Each classifier section has similar structure
#and variable names, so running them all at once may lead to odd behavior




##SKlearn L1 Penalized Logistic Regression##
############################################

#Sklearn requires arrays of floats
sk_train_X=Matrix(train_X)
sk_test_X=Matrix(test_X)
sk_train_y=float(recode(unwrap.(train_y), "O"=>1, "B"=>-1))
sk_test_y=float(recode(unwrap.(test_y), "O"=>1, "B"=>-1))

#Load and fit
sk_model=LogisticRegressionCV(penalty="l1",solver="liblinear")
sk_model.fit(sk_train_X,sk_train_y)

#Predict
y_pred=sk_model.predict(sk_test_X)

#test
cMat=MLJ.ConfusionMatrix()(y_pred,sk_test_y)


#Selecting coeffients

coef=sk_model.coef_ #Ith element is the coefficient for the feature in the i^th column of sk_train_X and train_X

cutoff=.1 #throw out features with coeffcients less than cutoff

ind=vec(abs.(coef) .> cutoff) #need to convert BitMatrix to BitVector to use in indexing

train_X[:,ind] #remaining feeatures




##SKlearn RidgeClassifierCV##
#############################

#Sklearn requites arrays of floats
sk_train_X=Matrix(train_X)
sk_test_X=Matrix(test_X)
sk_train_y=float(recode(unwrap.(train_y), "O"=>1, "B"=>-1))
sk_test_y=float(recode(unwrap.(test_y), "O"=>1, "B"=>-1))

#Load and fit
sk_model=RidgeClassifierCV()
sk_model.fit(sk_train_X,sk_train_y)

#Predict
y_pred=sk_model.predict(sk_test_X)

#test
cMat=MLJ.ConfusionMatrix()(y_pred,sk_test_y)


#Selecting coefficients

coef=sk_model.coef_ #i^th element is the coefficient for the feature in the i^th column of sk_train_X,train_X

cutoff=.4 #throw out features with coeffcients less than cutoff

ind=vec(abs.(coef) .> cutoff) #need to convert BitMatrix to BitVector to use in indexing

train_X[:,ind] #remaining feeatures




##MLJ L1 Penalized Logistic Classifier Example##
################################################

model = MLJLinearModels.LogisticClassifier( penalty=:l1) 

#params
l,u = 0, 1 #range of lambdas to search in hyperparameter search
num_models=50  #number of models to fit in hyperparameter search
num_folds=4 #number of elements in partition of training data for cross validation

lambda_range = range(model, :lambda, lower=l, upper=u) #Defining range object for hyperparameter tuning

self_tuning_model= MLJ.TunedModel(model=model,
    resampling = MLJ.CV(nfolds=num_folds, rng=1234),
    tuning = MLJ.RandomSearch(), #change to Grid() for random search
    range = lambda_range,
    n=num_models);

mach = MLJ.machine(self_tuning_model, train_X,train_y)

MLJ.fit!(mach) #Defaults to minimizing LogLoss during training


#Performance evaluation
pred_y=MLJ.predict_mode(mach,test_X) 
cMat=MLJ.ConfusionMatrix()(pred_y,test_y) #confusion matrix




##MLJ Lasso "Classifier" Example##
##################################

model =MLJLinearModels.LassoRegressor()

#Regressors require data to have scitype Continuous, here we change classes from a categorical array to an array of floats -1.0 and 1.0
reg_train_y=float(recode(train_y, "O"=>1, "B"=>-1))
reg_test_y=float(recode(test_y, "O"=>1, "B"=>-1))


#params
l,u = 0, 1 #range of lambdas to search in hyperparameter search
num_models=50  #number of models to fit in hyperparameter search
num_folds=4 #number of elements in partition of training data for cross validation

lambda_range = range(model, :lambda, lower=l, upper=u) #Defining range object for hyperparameter tuning

self_tuning_model= MLJ.TunedModel(model=model,
resampling = MLJ.CV(nfolds=num_folds, rng=1234),
tuning = MLJ.RandomSearch(), #change to Grid() for random search
range = lambda_range,
n=num_models);

mach = MLJ.machine(self_tuning_model, train_X,reg_train_y)

MLJ.fit!(mach)

#Performance evaluation
pred_y=MLJ.predict_mode(mach,test_X) 
pred_y=Int.(sign.(pred_y)) #round to -1 and 1, convert to int
cMat=MLJ.ConfusionMatrix()(pred_y,Int.(reg_test_y)) #confusion matrix




##MLJ Ridge Classifier Example##
################################

#For a ridge classifier we convert classes to -1.0 and 1.0 and then treat the problem as a regression problem for training.
#For class prediction we simply round outputs to -1 or 1


model =MLJLinearModels.RidgeRegressor()

#Regressors require data to have scitype Continuous, here we change classes from a categorical array to an array of floats -1.0 and 1.0
reg_train_y=float(recode(unwrap(train_y), "O"=>1, "B"=>-1))
reg_test_y=float(recode(unwrap(test_y), "O"=>1, "B"=>-1))

#params
l,u = 0, 1 #range of lambdas to search in hyperparameter search
num_models=50  #number of models to fit in hyperparameter search
num_folds=4 #number of elements in partition of training data for cross validation

lambda_range = range(model, :lambda, lower=l, upper=u) #Defining range object for hyperparameter tuning

self_tuning_model= MLJ.TunedModel(model=model,
resampling = MLJ.CV(nfolds=num_folds, rng=1234),
tuning = MLJ.RandomSearch(), #change to Grid() for random search
range = lambda_range,
n=num_models);

mach = MLJ.machine(self_tuning_model, train_X,reg_train_y)

MLJ.fit!(mach)

#Performance evaluation
pred_y=MLJ.predict_mode(mach,test_X) 
pred_y=Int.(sign.(pred_y)) #round to -1 and 1, convert to int
cMat=MLJ.ConfusionMatrix()(pred_y,Int.(reg_test_y)) #confusion matrix




##MLJ Elastic Net Classifier Example##
##############################

#For an Elastic Net Classifier we convert classes to -1.0 and 1.0 and then treat the problem as a regression problem for training.
#For class prediction we simply round outputs to -1 or 1


model =MLJLinearModels.ElasticNetRegressor()

#Regressors require data to have scitype Continuous, here we change from a categorical array to an array of floats -1.0 and 1.0
reg_train_y=float(recode(unwrap.(train_y), "O"=>1, "B"=>-1))
reg_test_y=float(recode(unwrap.(test_y), "O"=>1, "B"=>-1))

#params
l,u = 0, 1 #range of lambdas to search in hyperparameter search
num_models=50  #number of models to fit in hyperparameter search
num_folds=4 #number of elements in partition of training data for cross validation

#Defining range objects for hyperparameter tuning
lambda_range = range(model, :lambda, lower=l, upper=u)
gamma_range = range(model, :gamma, lower=l, upper=u)  

self_tuning_model= MLJ.TunedModel(model=model,
resampling = MLJ.CV(nfolds=num_folds, rng=1234),
tuning = MLJ.RandomSearch(), #change to Grid() for random search
range = [lambda_range, gamma_range],
n=num_models);

mach = MLJ.machine(self_tuning_model, train_X,reg_train_y)

MLJ.fit!(mach)

#Performance evaluation
pred_y=MLJ.predict_mode(mach,test_X) 
pred_y=CategoricalArray(Int.(sign.(pred_y))) #round to -1 and 1, convert to int
cMat=MLJ.ConfusionMatrix()(pred_y,Int.(reg_test_y)) #confusion matrix




####Selecting significant variables using MLJLinearModels##
###########################################################

params=MLJ.fitted_params(mach)
best_coefs=params.best_fitted_params.coefs #coefs of model with lowest log_loss

sig_factor_names=[] #store names of significant factors
cut_off=.1 #throw out factors with coeffient whose absolute value is less than cut_off

for pair in best_coefs

    col_name=pair[1]
    coef_val=pair[2]

    if abs(coef_val) > cut_off
        push!(sig_factor_names,col_name)
    end
end




##Some model evaluation Methods###
#################################

ypred=MLJ.predict_mode(mach,test_X) #will make prediction using model with lowest LogLoss

cMat=MLJ.ConfusionMatrix()(y_pred,Int.(reg_test_y)) #confusion matrix

entry=MLJ.report(mach).best_history_entry #report on best model 

best_log_loss=entry[:measurement]  #log_loss of best model

#Other performance measurements
MLJ.accuracy(y_pred,Int.(reg_test_y))
MLJ.f1score(y_pred,Int.(reg_test_y))





