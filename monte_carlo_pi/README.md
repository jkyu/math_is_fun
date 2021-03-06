# Monte Carlo simulation to determine the value of π.

A monte carlo simulation is implemented here with a neat animation showing convergence to π. 

## Background
A circle of radius r is inscribed in a square of side length 2r and area 4r^2.

X and Y are independent uniform random variables on [0, 2r].
This means that picking an x-coordinate at random says nothing about the y-coordinate in the (x, y) pair. 
This also means that there isn't any location on the line segment [0, 2r] that is favored over any other -- a point picked at random is equally likely to land anywhere in the interval. 

With probablility P = 1, the sampled (x, y) coordinate lands in the square, but how often does it land inside the circle versus outside of it? 

We know that the square has area 4r<sup>2</sup> and that the circle has an area proportional to r<sup>2</sup>. 
A<sub>circle</sub> = πr<sup>2</sup>, where π is that unknown proportionality constant.

Suppose we sample n points (x, y) from the joint distribution (X, Y). 
In the limit of large N, we expect that m number of points will land inside the circle out of the total n points. 
If we sample so many points that we completely black out the square, we will find that the ratio m/n is exactly the ratio of the areas of the inscribed circle to the square:
m/n = (πr<sup>2</sup>)/(4r<sup>2</sup>) = π/4

Then we will find that π will converge to π ≈ 4 * m/n as we increase our sample size.
