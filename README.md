
# PCA Background: Covariance Matrix and Eigendecomposition

## Introduction
In this lesson, we shall look at some background concepts required to understand how PCA works, with required mathematical formulas and python implementation. We shall look at covariance matrix, Eigen decomposition and will work with required numpy functions. 

## Objectives
You will be able to:
- Understand covariance matrix calculation with implementation in numpy
- Understand and explain Eigendecomspoistion with its basic characteristics
- Explain the role of eigenvectors and eigenvalues in eigendecomposition
- Decompose and re-construct a matrix using eigendecomposition

## Covariance

We have looked into correlation and covariance as measures to calculate how one random variable changes with respect to another. Covariance is always measured between 2 dimensions (variables).

> __If we calculate the covariance between a dimension and itself, we get the variance.__

So, if we a 3-dimensional data set (x, y, z), then we can measure the __covariance__ between the x and y dimensions, the y and z dimensions, and the x and z dimensions. Measuring the covariance between x and x, or y and y, or z and z would give us the __variance__ of the x, y and z dimensions respectively.

The formula for covariance is give as: 

$$cov(X,Y) = \frac{\sum_i^n(X_i -\mu_X)(Y_i - \mu_Y)}{n-1}$$

## The Covariance Matrix

We know that covariance is always measured between 2 dimensions __only__. If we have a data set with more than 2 dimensions, there is more than one covariance measurement that can be calculated. For example, from a 3 dimensional data set (x, y, z) you could calculate __cov(x,y), cov(x,z),__ and __cov(y,z)__. 

For an -dimensional data set, we can calculate $ \frac{n!}{(n-2)! * 2}$different covariance values.

A useful way to get all the possible covariance values between all the different dimensions is to calculate them all and put them in a matrix. The covariance matrix for a set of data with n dimensions would be:

$$C^{n x n} = (c_{i,j}, c_{i,j} = cov(Dim_i, Dim_j))$$

where $C^{m x n}$ is a matrix with $n$ rows and $n$ columns, and $Dim_x$ is the $i$th dimension.

So if we have an n-dimensional data set, then the matrix has n rows and n columns (square matrix) and each entry in the matrix is the result of calculating the covariance between two separate dimensions as shown below:

<img src="covmat.png" width=350>


- Down the main diagonal, we can see that the covariance value is between one of the dimensions and itself. These are the variances for that dimension.

- Since $cov(a,b) = cov(b,a)$, the matrix is symmetrical about the main diagonal.

### Calculate Covariance matrix in Numpy

In numpy, we can calculate the covariance of a given matrix using `np.cov()` function,  as shown below:

```python
# Covariance Matrix 
import numpy as np
X = np.array([ [0.1, 0.3, 0.4, 0.8, 0.9],
               [3.2, 2.4, 2.4, 0.1, 5.5],
               [10., 8.2, 4.3, 2.6, 0.9]
             ])
print( np.cov(X) )
```


```python
# Code here 
```

    [[ 0.115   0.0575 -1.2325]
     [ 0.0575  3.757  -0.8775]
     [-1.2325 -0.8775 14.525 ]]


The diagonal elements, $C_{ii}$ are the variances in the variables $x_i$ assuming N−1 degrees of freedom:

```python
print(np.var(X, axis=1, ddof=1))
```


```python
# Code here 
```

    [ 0.115  3.757 14.525]


## Eigendecomposition

The eigendecomposition is one form of matrix decomposition. Decomposing a matrix means that we want to find a product of matrices that is equal to the initial matrix. In the case of the eigendecomposition, we decompose the initial matrix into the product of its __eigenvectors__ and __eigenvalues__.
 
A vector $v$ is an __eigenvector__ of a __square__ matrix $A$ if it satisfies the following equation:

$$A.v = \lambda.v$$

Here, __lambda__ ($\lambda$) is the represents the __eigenvalue__ scalar.

> A matrix can have one eigenvector and eigenvalue for each dimension of the parent matrix. 

Also , remember that not all square matrices can be decomposed into eigenvectors and eigenvalues, and some can only be decomposed in a way that requires complex numbers. __The parent matrix can be shown to be a product of the eigenvectors and eigenvalues.__

$$A = Q . diag(V) . Q^-1$$

$Q$ is a matrix comprised of the eigenvectors, $diag(V)$ is a diagonal matrix comprised of the __eigenvalues__ along the diagonal, and $Q^-1$ is the inverse of the matrix comprised of the eigenvectors.

A decomposition operation breaks down a matrix into constituent parts to make certain operations on the matrix easier to perform. Eigendecomposition is used as an element to simplify the calculation of other more complex matrix operations.

### Eigenvectors and Eigenvalues

__Eigenvectors__ are unit vectors, with length or magnitude is equal to 1.0. They are often referred as right vectors, which simply means a column vector (as opposed to a row vector or a left vector). Imagine a transformation matrix that, when multiplied on the left, reflected vectors in the line $y=x$. We can see that if there were a vector that lay on the line $y=x$, it’s reflection it itself. This vector (and all multiples of it), would be an eigenvector of that transformation matrix.


![](eig1.png)

__Eigenvalues__ are coefficients applied to eigenvectors that give the vectors their length or magnitude. For example, a negative eigenvalue may reverse the direction of the eigenvector as part of scaling it. Eigenvalues are closely related to eigenvectors.

A matrix that has only positive eigenvalues is referred to as a __positive definite matrix__, whereas if the eigenvalues are all negative, it is referred to as a __negative definite matrix__.

Decomposing a matrix in terms of its eigenvalues and its eigenvectors gives valuable insights into the properties of the matrix. Certain matrix calculations, like computing the power of the matrix, become much easier when we use the eigendecomposition of the matrix. The eigendecomposition can be calculated in NumPy using the `eig()` function.

The example below first defines a 3×3 square matrix. The eigendecomposition is calculated on the matrix returning the eigenvalues and eigenvectors using `eig()` method.

```python
# eigendecomposition
from numpy import array
from numpy.linalg import eig
# define matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# calculate eigendecomposition
values, vectors = eig(A)
print(values)
print(vectors)
```


```python
# Code here 
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    [ 1.61168440e+01 -1.11684397e+00 -9.75918483e-16]
    [[-0.23197069 -0.78583024  0.40824829]
     [-0.52532209 -0.08675134 -0.81649658]
     [-0.8186735   0.61232756  0.40824829]]


## Testing an Eigenvector

Above, The eigenvectors are returned as a matrix with the same dimensions as the parent matrix (3x3), where each column is an eigenvector, e.g. the first eigenvector is vectors[:,0]. Eigenvalues are returned as a list, where value indices in the returned array are paired with eigenvectors by column index, e.g. the first eigenvalue at values[0] is paired with the first eigenvector at vectors[: 0].

We will now test whether the first vector and value are in fact an eigenvalue and eigenvector for the matrix. We know they are, but it is a good exercise.

```python
# confirm first eigenvector
B = A.dot(vectors[:, 0])
print(B)
C = vectors[:, 0] * values[0]
print(C)
```


```python
# Code here 
```

    [ -3.73863537  -8.46653421 -13.19443305]
    [ -3.73863537  -8.46653421 -13.19443305]


### Reconstruct Original Matrix

We can reverse the process and reconstruct the original matrix given only the eigenvectors and eigenvalues.

First, the list of eigenvectors must be converted into a matrix, where each vector becomes a row. The eigenvalues need to be arranged into a diagonal matrix. The NumPy `diag()` function can be used for this. Next, we need to calculate the inverse of the eigenvector matrix, which we can achieve with the `inv()` function. Finally, these elements need to be multiplied together with the `dot()` function.

```python
from numpy.linalg import inv
# create matrix from eigenvectors
Q = vectors
# create inverse of eigenvectors matrix
R = inv(Q)
# create diagonal matrix from eigenvalues
L = np.diag(values)
# reconstruct the original matrix
B = Q.dot(L).dot(R)
print(B)
```


```python
# Code here 
```

    [[1. 2. 3.]
     [4. 5. 6.]
     [7. 8. 9.]]


## Further Resources

Above description provides an overview of eigendecomposition and covariance matrix. You are encouraged to visit following resources to get a deep dive into underlying mathematics for above equations. 

[Variance- Covariance Matrix](https://stattrek.com/matrix-algebra/covariance-matrix.aspx)

[The Eigen-Decomposition:
Eigenvalues and Eigenvectors](https://www.utdallas.edu/~herve/Abdi-EVD2007-pretty.pdf)

[Eigen Decomposition Visually Explained](http://setosa.io/ev/eigenvectors-and-eigenvalues/)

## Summary 

In this lesson, we looked at calculating covariance matrix for a given matrix. We also looked at Eigen decomposition and its implementation in python. We can now go ahead and use these skills to apply principle component analysis for a given multidimensional dataset using these skills.  
