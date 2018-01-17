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


#print Train_Data
#print Train_Label

# Defining Variables....Note you have to mention trainable = True in W
# Else it wont know which varible is to be updated while minimising the loss function

X = tf.get_variable( initializer= tf.constant( Train_Data ), dtype = 'float32', name= "X")
Y = tf.get_variable( initializer= tf.constant( Train_Label ), dtype='int32', name = "Y" )
W = tf.get_variable( initializer= tf.constant( [[1.0,1.0]] ), dtype='float32', trainable=True, name = "W" )

# Sigmoid Function
Dot_pdt = tf.matmul( X, tf.transpose(W) )
Sigmoid = tf.get_variable( initializer= 1/( 1 + tf.exp(-Dot_pdt)), name = "mu" )

# Loss Function
Loss_fn = tf.get_variable( initializer = tf.reduce_mean( tf.to_float(Y)*tf.log(Sigmoid) + (1-tf.to_float(Y))*tf.log(1-Sigmoid) ), name="loss" )

# Training
optimiser = tf.train.GradientDescentOptimizer(learning_rate=1.1).minimize(Loss_fn)

# Evaluating
with tf.Session() as sess:

    writer = tf.summary.FileWriter('/home/divyat/Desktop/Workspace/Machine Learning/TensorFlow/Degub', graph=tf.get_default_graph())

    sess.run(X.initializer)
    print sess.run(X)

    sess.run(Y.initializer)
    print sess.run(Y)

    sess.run(W.initializer)
    print sess.run(W)

    sess.run(Dot_pdt)

    sess.run(Sigmoid.initializer)
    print sess.run(Sigmoid)

    sess.run(Loss_fn.initializer)
    print sess.run(Loss_fn)

    #print sess.run(tf.to_float(Y)*tf.log(Sigmoid) + (1-tf.to_float(Y))*tf.log(1-Sigmoid))

    for _ in range(1000):
        print sess.run(optimiser, feed_dict={X: Train_Data, Y: Train_Label})
        print sess.run(W), "\n"
        print sess.run(Loss_fn), "\n"

    print sess.run(Loss_fn)
    print sess.run(Sigmoid)

#writer.close()