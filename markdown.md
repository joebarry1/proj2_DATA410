# Project 2

## Locally Weighted Regression

Locally weighted regression (LWR) is a regression based form of modelling. Its main feature is that it is able to fit a complex relationship between a target and its feature(s) while maintaining a smooth curve.

Mathematically, we can understand LWR by first looking at the principle equation of linear regression:

<img src="https://render.githubusercontent.com/render/math?math=$\large y = X\cdot\beta \pm \sigma\epsilon">

We can weight this equation by some *W* by multiplying it to both sides:

<img src="https://render.githubusercontent.com/render/math?math=\large Wy = WX\cdot\beta +\sigma W\epsilon">

The usual weight used in LOESS is the tri cubic weight kernel. The weight on a given *x* using this kernel is: 

<img src="https://render.githubusercontent.com/render/math?math=w(x) = (1-\abs{d}^3)^3">

