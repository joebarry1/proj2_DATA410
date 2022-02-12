# Project 2

## Locally Weighted Regression

Locally weighted regression (LWR) is a regression based form of modelling. Its main feature is that it is able to fit a complex relationship between a target and its feature(s) while maintaining a smooth curve.

Mathematically, we can understand LWR by first looking at the principle equation of linear regression:

$$\large y = X\cdot\beta +\sigma\epsilon $$

We can weight this equation by some $W$ by multiplying it to both sides:

$$\large Wy = WX\cdot\beta +\sigma W\epsilon $$

The usual weight used in LOESS is the tri cubic weight kernel. The weight on a given $x$ using this kernel is: 

$$ w(x) = (1-\abs{d}^3)^3 $$
