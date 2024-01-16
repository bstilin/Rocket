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



    store=[]; B=zeros(size(A))

    for i in 1:size(A)[1]

        average=mean(A[i,:])
        sd=std(A[i,:])

        #Avoiding dividing by zero

        if sd != 0
            B[i,:]=((A[i,:].-average)/sd)
        else sd == 0
            B[i,:]=((A[i,:].-average))
        end

        push!(store,(average=average, sd=sd))
    end

    return B,store
end


"
Partitions a vector into sections of length num_points and stacks them vertically.
The remainder is discarded from the end of the vector if the time series cannot be 
divded exactly (see example).

### Parameters
1. series : Vector
    - A single time series
2. num_points : Integer
    - The number of points in each chunk

### Returns 
    -An array with num_points columns 

### Example 

A=[1, 2, 3, 4, 5 ,6, 7, 8]

julia> series_split(A,3)

2x3 Matrix{Int64}: \\
 1  2  3  \\
 4  5  6
"
function series_split(series :: Array, num_points :: Integer)

    data_array=[]

    for chunk in Iterators.partition(series,num_points)

        if length(chunk)==num_points
            push!(data_array, chunk)
        end
    end

    data_array=mapreduce(permutedims,vcat,data_array)

    return(data_array)

end


"
Performs a test train split for a classification problem. 

###Parameters

1. x : Array
    - An array where each row is assumed to be an observation
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

    train_x=x[train_ind,:]; train_y=y[train_ind]
    test_x=x[test_ind,:]; test_y=y[test_ind]

    return train_x, train_y, test_x, test_y

end