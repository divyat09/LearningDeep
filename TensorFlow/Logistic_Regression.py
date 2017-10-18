import tensorflow as tf
import numpy as np

# Training Data
Train_Data= []
Train_Label= []  # Labels

with open("DataSet.txt") as f:
    Data = f.readlines()

# Loading Data
for item in Data:
    
    item = item.replace("\n","")
    DataPoint = []
    DataPoint.append( float(item.split("\t")[0]) )
    DataPoint.append( float(item.split("\t")[1]) )
    
    Train_Data.append(DataPoint)
    Train_Label.append( int(item.split("\t")[2]) )

Train_Data = np.array(Train_Data)
Train_Label = np.array(Train_Label)

# Defining Variables....Note you have to mention trainable = True in W
# Else it wont know which varible is to be updated while minimising the loss function

X = tf.get_variable( initializer= tf.zeros_initializer(), shape = (10,2), dtype = 'float32', name= "X")
Y = tf.get_variable( initializer= tf.zeros_initializer(), shape = (10), dtype='int32', name = "Y" )
W = tf.get_variable( initializer= tf.constant( [[1.9,2.7]] ), dtype='float32', trainable=True, name = "W" )

# Sigmoid Function
Dot_pdt = tf.matmul( X, tf.transpose(W) )
Sigmoid = tf.get_variable( initializer= 1/( 1 + tf.exp(-Dot_pdt)), name = "mu" )

# Loss Function
Loss_fn = tf.get_variable( initializer = tf.reduce_sum( tf.to_float(Y)*tf.log(Sigmoid) + (1-tf.to_float(Y))*tf.log(1-Sigmoid) ), name="loss" )

# Training
optimiser = tf.train.GradientDescentOptimizer(learning_rate=1.1).minimize(Loss_fn)

# Evaluating
sess = tf.Session()

print sess.run(tf.shape(X))
print sess.run(tf.shape(W))
print sess.run(tf.shape(Dot_pdt))
print "\n"

sess.run(X.initializer)
sess.run(X)

sess.run(W.initializer)
sess.run(W)

sess.run(Dot_pdt)

sess.run(Sigmoid.initializer)
sess.run(Sigmoid)

sess.run(Loss_fn.initializer)
sess.run(Loss_fn)

for _ in range(1000):
    sess.run(optimiser, feed_dict={X: Train_Data, Y: Train_Label})
    print sess.run(W), "\n"
    print sess.run(Loss_fn), "\n"

sess.close()