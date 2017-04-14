import tensorflow as tf

def conv(indata,ksize,strides,padding,name):
    W = tf.get_variable(name+"_weights", shape=ksize,initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+"_bias", shape=[ksize[-1]])
    return b + tf.nn.conv2d(indata,W,strides=strides,padding=padding)
def batchnorm(inp,name):
    ksize = inp.get_shape().as_list()
    ksize = [ksize[-1]]
    mean,variance = tf.nn.moments(inp,[0,1,2],name=name+'_moments')
    scale = tf.get_variable(name+"_scale", shape=ksize)#,initializer=tf.contrib.layers.variance_scaling_initializer())
    offset = tf.get_variable(name+"_offset", shape=ksize)#,initializer=tf.contrib.layers.variance_scaling_initializer())
    return tf.nn.batch_normalization(inp,mean=mean,variance=variance,scale=scale,offset=offset,variance_epsilon=1e-5)

def getcnnfeature(im):
    conv1 = conv(im,ksize=[3,3,3,64],strides=[1,1,1,1],padding='SAME',name='conv1')
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = conv(pool1,ksize=[3,3,64,128],strides=[1,1,1,1],padding='SAME',name='conv2')
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3 = conv(pool2,ksize=[3,3,128,256],strides=[1,1,1,1],padding='SAME',name='conv3')
    conv4 = conv(conv3,ksize=[3,3,256,256],strides=[1,1,1,1],padding='SAME',name='conv4')
    pool4 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv5 = conv(pool4,ksize=[3,3,256,512],strides=[1,1,1,1],padding='SAME',name='conv5')
    conv5 = batchnorm(conv5,name='bn5')
    conv6 = conv(conv5,ksize=[3,3,512,512],strides=[1,1,1,1],padding='SAME',name='conv6')
    conv6 = batchnorm(conv6,name='bn6')
    pool6 = tf.nn.max_pool(conv6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv7 = conv(pool6,ksize=[3,3,512,512],strides=[1,1,1,1],padding='SAME',name='conv7')
    #conv8 = conv(conv7,ksize=[2,2,512,2048],strides=[1,2,1,1],padding='VALID',name='conv8')
   
    return conv7

def getcnnlogit(im,outnum):
    fea = getcnnfeature(im)
    fea = tf.transpose(fea,[0,2,1,3])
    feashape = fea.get_shape().as_list()
    print feashape
    feadim = feashape[-1]*feashape[-2]*feashape[-3]
    fea = tf.reshape(fea,[feashape[0],-1])
    W = tf.get_variable("logit_weights", shape=[feadim,outnum],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("logit_bias", shape=[outnum],initializer=tf.contrib.layers.xavier_initializer())
    return tf.matmul(fea,W)+b
