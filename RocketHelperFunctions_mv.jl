using Statistics
using StatsBase
using Random

"Normalizes the rows of the input matrix. Each row has its mean subtracted and is divided by its standard deviation.
 The resulting rows have mean 0 and standard deviation 1.

###Parameters
1. A : Array  
    - A is an n x m array

### Returns
    -A tuple (B,store) where B is an n x m array whose j^th row is the normalized j^th row of A
    and store is vector whose j^th entry is a named tuple containing the average and standard deviation
    of the j^th row of A"
function normalize_data_matrix(A :: Array)
#Normalize over all dimensions


    store=[]; B=zeros(size(A))

    for i in 1:size(A)[1]

        average=mean(A[i,:])
        sd=std(A[i,:])

        #Avoiding dividing by zero

        if sd != 0
            B[i,:]=(A[i,:].-average)/sd
        else sd == 0
            B[i,:]=(A[i,:].-average)
        end

        push!(store,(average=average, sd=sd))
    end

    return B,store
end



"
Performs a test train split for a classification problem (multivariate). 

###Parameters

1. x : Array
    - An array where each row is assumed to be an observation with d dimensions and t time points.
2. y : Vector
    - A vector containing the classes of the observations
3. p : Float64
    - The fraction of data to be used for training

### Returns 
    - A tuple (train_x, train_y, test_x, test_y) where train_x is the input training data and train_y
    is the training classes. test_x, test_y defined similarly.
"
function test_train_split(x :: Array, y :: Vector, p :: Float64)

    num_samples=size(x)[1]

    num_train=Int(floor(p*num_samples));

    ind=shuffle(1:num_samples); train_ind=ind[1:num_train]; test_ind=ind[num_train+1:end]

    train_x=x[train_ind,:,:]; train_y=y[train_ind]
    test_x=x[test_ind,:,:]; test_y=y[test_ind]

    return train_x, train_y, test_x, test_y, [train_ind, test_ind]

end

"""
Given a multivariate data matrix X, normalize each sample in each dimension independently
###Parameters

    X : Array
    - A three-dimensional time series array structured as [samples, dimensions, timepoints]

### Returns 
    - A tuple (T, St) where T is an n x d x m array whose j^th row contains the normalized j^th row of X in each dimension
    and St is vector whose j^th entry is a named tuple containing the average and standard deviation
    in each dimension of the j^th row of X
"""
function normalize_data_matrix_mv(X::Array)
    T = copy(X)
    St = []
    for i in 1:size(X)[2]
        T[:,i,:], stor = normalize_data_matrix(T[:,i,:])
        push!(St, stor)
    end
    return T, St
end
