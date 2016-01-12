import os, sys
try:
    import theano
    import theano.tensor as T
    print 'OK::theano imported successfully'
    import numpy as np
    import matplotlib.pyplot as plt

except NameError as e:
    print e

###
#Least-sqr Regression
###

X = np.array([60, 69, 66, 64, 54, 67, 59, 65, 63])
Y = np.array([136, 198, 194, 140, 93, 172, 116, 174, 145])
print (len(X), len(Y))
Y_bar = sum(Y)/ len(Y)
X_bar = sum(X)/len(X)

A = X - X_bar
B = Y - Y_bar

topb = sum(A * B)
print 'top b --> {}'.format(topb)


x = T.vector('x')
y = T.vector('y')

xx = T.vector('xx')

z = x * x
f = theano.function([x], z)
ts = f(A)
bb = sum(ts)

slope_b = topb/bb
intcp_b = Y_bar - (slope_b * X_bar)
print 'slope b --> {}'.format(slope_b)
print 'intercept b --> {}'.format(intcp_b)

g = (slope_b * xx) + intcp_b
gf = theano.function([xx], g)
yy = gf(X)

plt.plot(X, Y, 'ro')
plt.plot(X, yy, 'b')
plt.axis([0, 100, 0, 200])
plt.ylabel("Weight (g) \n Vipera bertis")
plt.xlabel('Length (cm)')
plt.show()
sys.exit()
