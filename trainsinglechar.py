import tensorflow as tf
import numpy as np
import cv2
from cnn import getcnnfeature,getcnnlogit

def getdata(datalst,batchsize,start,ab_size,imsize):
    label = np.zeros([batchsize,ab_size])
    data = np.zeros([batchsize,]+imsize)
    for i in range(batchsize):
        f,l = datalst[i+start].split()
        im = cv2.imread("../../danziimg/dataset/"+f)
        im = cv2.resize( im,(32,32) )
        #for c in range(3):
        data[i,:,:,:] = im[:,:,:]/255.0
        cc = (int(l) - 1)%ab_size
        label[i,cc] = 1
    return data,label

if __name__=='__main__':
    ab_size = 3817
    bs = 32
    imsize = [32,32,3]
    im = tf.placeholder(tf.float32,shape=[bs,]+imsize)
    label = tf.placeholder(tf.float32,shape=[bs,ab_size])

    logit = getcnnlogit(im,ab_size)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
   
    correct_prediction = tf.equal(tf.argmax(logit,1), tf.argmax(label,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    trainlst = []
    for line in open("datalist/train.lst"):
        trainlst.append(line)
    testlst = []
    for line in open("datalist/test.lst"):
        testlst.append(line)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,tf.train.latest_checkpoint('ckpt32/'))
        for e in range(5):
            for i in range(len(trainlst)/bs):
                d,l = getdata(trainlst,bs,i*bs,ab_size,imsize)
                loss,_ = sess.run([cross_entropy,opt],{im:d,label:l})
                print "epoch:",e,"iter:",i,"loss:",loss
                if i%1000 == 0:
                    accuracy = 0
                    for j in range(len(testlst)/bs):
                        d,l = getdata(testlst,bs,j*bs,ab_size,imsize)
                        accuracy += sess.run(acc,{im:d,label:l})
                    accuracy /= len(testlst)/bs
                    print "iter:",j,"accuracy:",accuracy
                    saver.save(sess,"ckpt32/danzi.ckpt",e*len(trainlst)+i)
        saver.save(sess,"ckpt32/final.ckpt")
