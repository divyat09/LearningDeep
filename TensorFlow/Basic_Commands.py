# Some basic stuff

v1 = tf.Variable([12,10,20], dtype=tf.float32, name= 'vector1')
v2 = tf.Variable([32,20,10], dtype=tf.float32, name= 'vector1')

r1 = tf.multiply(v1, v2, name = 'mul') # Tensor as result of element wise multiplication of v1 and v2
r2 = tf.reduce_sum(r1)                 # Sum of elements of a Tensor
r3 = tf.tensordot(v1, v2, axes = 1)    # Dot product of 2 1-D Tensors

sess = tf.Session()

# TO initialise all the Tensor Flow variables you have to initalise them as follows
# Without running sess.run(init) any attempt to evalute expressions that use Variables would result in error
init = tf.global_variables_initializer()
sess.run(init)

r1_val = sess.run(r1)
r2_val = sess.run(r2)
r3_val = sess.run(r3)

print r1_val
print r2_val
print r2_val == r3_val # Dot product along axis 1 is same as sum of tensor pdt for 1-D tensors

# Sigmoid Function for Logistic Regression

# You need to define x and w as variable else you cannot use them for training model
x = tf.random_normal([10,2], name = 'x')
w = tf.random_normal([10,2], name = 'w')

# Dot Product of tensor along an axis
dot_product = tf.tensordot( w, x, axes=1 )

# Sigmoid Function
sig_x = 1/(1 + tf.exp(-dot_product))

# Running the computational graph using a session
sess = tf.Session()
x_val = sess.run(x)
w_val = sess.run(w)
dotpdt_val = sess.run(dot_product)
sig_val = sess.run(sig_x)

print x_val
print w_val
print dotpdt_val
print sig_val
