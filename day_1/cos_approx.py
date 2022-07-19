#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Ahmed Issawi'
__email__ = 'aissawi@unb.ca'

from math import factorial
from math import pi

def cos_approx(x, accuracy=10):
    """
    The Maclaurin series expansion for cos(x)
    - X  integer number 
    Independent variable 
    - Accuracy integer
    series of numbers identify the range we will substitute in the equation
    
    Our result is the Sum of the list we created from the calculation of the body equation 
    which is the range of the cosin values. 
    """
    
    return sum([ (((-1)**n) * x ** (2*n)) / (factorial(2 * n)) for n in range(accuracy)]) 

#Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))