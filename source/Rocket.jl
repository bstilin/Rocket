"Class definition and functions for implementing the Rocket transform. This implementaiton does NOT perform standardization 
of the input data. Further, it assumes that the time series are all of the same length and do not have any missing data" 

using Statistics
using StatsBase
using Random


Base.@kwdef struct Kernel

    length :: Int32
    weights :: Vector{Float64}
    bias :: Float64
    dilation :: Int32
    padding :: Int32
    name :: Symbol

end

"Used to generate a list of Rocket kernels

### Parameters
1. series_length : int
    - Length of time series
2. num_kernels : int
    - Number of kernels to generate


### Returns
- A vector of kernel objects of length num_kernel"
function generate_kernels(series_length :: Integer, num_kernels :: Integer)
    
    Kernels=[]

    #Generate Random Lengths
    candidate_lengths = Array{Int32}([7,9,11])
    lengths=rand(candidate_lengths, num_kernels)

    for i in range(1,stop=num_kernels)

        _name = Symbol("K",i)
        _length = lengths[i]

        _weights = randn(_length)
        _weights = _weights .- mean(_weights) 

        _bias = rand()*2-1 

        dilation = 2^(rand() * log2( (series_length-1) / (_length-1) ))
        _dilation = floor(Int32,dilation)

        if rand([1,2])==1
            padding= fld( (_length-1) * _dilation , 2 )
        else 
            padding = 0
        end

        _padding=padding

        K=Kernel(length=_length, weights=_weights, bias=_bias, dilation=_dilation, padding=_padding, name=_name)
        push!(Kernels,K)

    end

    return Kernels
end

"Apply a kernel to a time series and output PPV and max

### Parameters
1. X : Array
    - A time series
2. K : Kernel
    - Kernel to apply

### Returns
- A tuple (max,PPV)"
function apply_kernel(X :: Union{Array,SubArray}, K :: Kernel)

    weights=K.weights; filter_length=K.length; bias=K.bias; dilation=K.dilation; padding=K.padding

    input_length=length(X)
    output_length= (input_length + (2 * padding)) - ((filter_length - 1) * dilation)

    _ppv=0
    _max=-1.e30

    End = (input_length + padding) - ((filter_length - 1) * dilation)
    for i in range(-padding+1,stop=End) #Iterate of index of left endpoint of kernel

        _sum=bias
        index=i

        for j in range(1,stop=filter_length) #Iterate over length of kernel for dot product

            if index > 0 && index <= input_length #Account for padding

                _sum = _sum +weights[j] * X[index]

            end

            index = index + dilation #accounts for dilation
        end

        if _sum > _max

            _max = _sum
        end

        if _sum > 0
            _ppv +=1
        end
    end
        
    return  _max,_ppv/output_length
end

"""Transforms a selection of time series into kernel space

### Parameters
1. X : Array
    - An n x m matrix containing n time series of length m
2. K : Vector
    - A vector of length p of kernel objects

### Returns
- A tuple (X_transform,max,PPV) where
1. X_transform is a n x (2*num_kernels) array where the first num_kernels entries of the i^th row are the max values obtained from 
applying the kernels to the i^th row of X_train and the next num_kernels entries are the PPV's obtained. The i^th row is the
representation of the i^th row of X in kernel space
2. max is an n x p array such that its i,j entry is the max obtained from applying the j^th kernel to the i^th row of X.
3. PPV is defined similarly as max""" 
function Rocket_transform(X :: Union{SubArray,Array}, K :: Vector)


    num_examples=size(X)[1]
    num_kernels=size(K)[1]

    ppv = zeros(Float64,num_examples,num_kernels)
    max = zeros(Float64,num_examples,num_kernels)

    
    Threads.@threads for i in 1:size(X)[1]
        for (j,kernel) in enumerate(K)
        
        Xview= @view X[i,:]
        max[i,j],ppv[i,j]=apply_kernel(Xview,kernel)

        end
    end

    X_transform=hcat(max,ppv)

    return X_transform,max,ppv
end
