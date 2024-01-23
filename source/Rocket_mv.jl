#Class definition and functions for implementing Rocket transform
"""
A kernel struct defined by:
    - length,
    - a rectangular array of weights (n_dim x length),
    - bias,
    - dilation,
    - padding,
    - identifier
"""
Base.@kwdef struct Kernel_mv

    length :: Int32
    weights :: Array{Float64}
    bias :: Float64
    dilation :: Int32
    padding :: Int32
    name :: Symbol

end


"""Used to generate a list of kernels

### Parameters
1. series_length : int
    - Length of time series
2. series_dim : int
    - Number of channels/dimensions for a time series
2. num_kernels : int
    - Number of kernels to generate

### Returns
- A vector of Kernel_mv objects of length num_kernel"""
function generate_kernels_mv(series_length :: Integer, series_dim :: Integer, num_kernels :: Integer)
    
    Kernels=[]

    #Generate Random Lengths
    candidate_lengths = Array{Int32}([7,9,11,13,15])
    lengths=rand(candidate_lengths, num_kernels)

    for i in range(1,stop=num_kernels)

        _name = Symbol("K",i)
        _length = lengths[i]

        _weights = randn(series_dim,_length)
        _weights = _weights .- mean(_weights) #check this on a per-dim level

        _bias = rand()*2-1 

        dilation = 2^(rand() * log2( (series_length-1) / (_length-1) ))
        _dilation = floor(Int32,dilation)

        if rand([1,2])==1
            padding= fld( (_length-1) * _dilation , 2 )
        else 
            padding = 0
        end

        _padding=padding

        K=Kernel_mv(length=_length, weights=_weights, bias=_bias, dilation=_dilation, padding=_padding, name=_name)
        push!(Kernels,K)

    end

    return Kernels
end


"""Apply a multivariate kernel to a time series and output PPV and average max (across all dimensions)
This adds correlation structure between dimensions of the kernel filter. 
PPV is calculated across all convolution outputs for all dimensions.

### Parameters
1. X : Array
    - A multivariate time series, with n_dim channels
2. K : Kernel
    - Multivariate Kernel to apply.

### Returns
- A tuple (avg_max over all kernel dimensions,PPV)"""
function apply_kernel_mv(X :: SubArray, K :: Kernel_mv)

    weights=K.weights; filter_length=K.length; bias=K.bias; dilation=K.dilation; padding=K.padding

    dims, input_length=size(X)
    output_length= (input_length + (2 * padding)) - ((filter_length - 1) * dilation)
    _ppv=0
    _max=-1.e30

    End = (input_length + padding) - ((filter_length - 1) * dilation)
    av_max = 0
    _max=-1.e30
    for dim in 1:dims
        _max = -1.e30

        for i in range(-padding+1,stop=End) #Iterate of index of left endpoint of kernel

            _sum=bias
            index=i

            for j in range(1,stop=filter_length) #Iterate over length of kernel for dot product

                if index > 0 && index <= input_length #Account for padding

                    _sum = _sum + weights[dim,j] * X[dim,index]

                end

                index = index + dilation #accounts for dilation
            end

            if _sum > _max

                _max = _sum
            end
            if _sum > 0
                _ppv += 1
            end
        end
        av_max += _max
    end 
    #Here, we choose the average maximum overlap between each kernel dimension
    #though this asymptotically becomes less useful as kernel dimension increases.
    #we also calculate the overall ppv of the combined kernels.
    #This becomes a big problem as the dimension of the kernel space increases.
    return  av_max/dims,_ppv/(output_length*dims)
end


"""Transforms a selection of time series into kernel space

### Parameters
1. X : Array
    - An n x m matrix containing n time series of length m
2. K : Vector
    - A vector of length p of multivariate kernel objects

### Returns
- A tuple (X_transform,avg_max,PPV) where
1. X_transform is a n x (2*num_kernels) array where the first num_kernels entries of the i^th row are the max values obtained from 
applying the kernels to the i^th row of X_train and the next num_kernels entries are the PPV's obtained. The i^th row is the
representation of the i^th row of X in kernel space
2. avg_max is an n x p array such that its i,j entry is the max obtained from applying the j^th kernel to the i^th row of X.
3. PPV is defined similarly as avg_max""" 
function Rocket_transform_mv(X :: Array, K :: Vector)


    num_examples=size(X)[1]
    num_kernels=size(K)[1]

    ppv = zeros(Float64,num_examples,num_kernels)
    max = zeros(Float64,num_examples,num_kernels)
    
    Threads.@threads for i in 1:size(X)[1]
        for (j,kernel) in enumerate(K)
            V = @view X[i,:,:]
            max[i,j],ppv[i,j]=apply_kernel_mv(V,kernel)

        end
    end

    X_transform=hcat(max,ppv)

    return X_transform,max,ppv
end