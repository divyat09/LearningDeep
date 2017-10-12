import numpy as np
import theano as th
import theano.tensor as T
from theano import function
from theano import pp
from theano import shared

x = T.dscalar('x')
y = T.dscalar('y')

z1 = x - y
z2 = x + y

v = T.dvector('vec')
z3 = v + v**2

# Creating a Theano function.....function( input, output )
f = function([x, y], [z1, z2])
g= function([v], z3)

# Prints the structure of the theano variable
print pp(x)
print pp(y)
print pp(z1)
print pp(z2)
print pp(z3)

# Computes the value of the
print f(2,3)
print g( np.array([1,3,2,5]))

# Finding sigmoid of the elements of a matrix

M = T.dmatrix('Mat')
sigmoid = 1/(1+T.exp(-M))
sig = function([M], sigmoid)

print pp(sigmoid)
print sig(np.array([[1,3],[5,4]]))

# Shared Variables

state = shared(2)
inc = T.iscalar('inc')
print state.get_value()

adder = function([inc], state, updates = [(state, state + inc)])
print state.get_value()
