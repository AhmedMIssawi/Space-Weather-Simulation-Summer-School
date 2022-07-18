#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Ahmed Issawi'
__email__ = 'aissawi@unb.ca'

from math import factorial
from math import pi
from scipy.integrate import quad


def cos_approx(x, accuracy=10):
    """
    
    """
    # powers = [n for n in range(accuracy)]
    
    # calculated_results = [ (((-1)**n) * x ** (2*n)) / (factorial(2 * n)) for n in range(powers)]
    
    
    # return sum(calculated_list)
    
    power = [n for n in range(accuracy)]
    upper = (-1) ** power
    lower = (2 * factorial(power))
    right = x ** (2 * power)
    result_BS = (upper * right) / lower
    calculated_list = []
    return sum(calculated_list) 



# Will only run if this is run from command line as opposed to imported
# if __name__ == '__main__':  # main code block
#     print("cos(0) = ", cos_approx(0))
#     print("cos(pi) = ", cos_approx(pi))
#     print("cos(2*pi) = ", cos_approx(2*pi))
#     print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
