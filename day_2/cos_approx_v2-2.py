# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:25:19 2022

@author: Ahmed Issawi
"""
import argparse
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


def parse_args():
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = \
                                   'My Example Code V2')
  # # in_scalar: scalar value, type float:
  parser.add_argument('-x', \
                    help = 'my scalar variable', \
                    type = float)
   #npts: scalar value, type integer, default 5:
  parser.add_argument('-accuracy',  help = 'another scalar (default = 10)', type = int, default = 10)
  # # do_this: scalar value, boolean, default false:
  # parser.add_argument('-do_this', \
  #                     help = 'Boolean to do something', \
  #                     action = 'store_true')
  # actually parse the data now:
  args = parser.parse_args()
  return args


#Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    # print("cos(0) = ", cos_approx(0))
    # print("cos(pi) = ", cos_approx(pi))
    # print("cos(2*pi) = ", cos_approx(2*pi))
    # print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    
    # parse the input arguments:
    args = parse_args()
    print(args)
    # grab the variable in_scalar (a scalar):
    x = args.x
    print(x)
    # grab the number of points (an integer, default 5):
    accuracy = args.accuracy
    print(accuracy)
    print(cos_approx(x,accuracy))
    
    # comparison value
    eta = 1.2e-2
    if cos_approx(x,accuracy) < cos_approx(x,accuracy) + eta and cos_approx(x, accuracy) > cos_approx(x,accuracy) - eta:
       print('lala')
    else:
        print('none')
        
        
           
