import random
import tensorflow as tf
import numpy as np
import cv2
from cnn import getcnnfeature,getcnnlogit
from PIL import Image

def getdata(datalst,batchsize,start,ab_size,imsize):
    label = np.zeros([batchsize,ab_size])
    data = np.zeros([batchsize,]+imsize)
    lengths = []
    idx = []
    vals = []
    for i in range(batchsize):
        labels = datalst[i+start].split()
        f = labels[0]
        im = Image.open("../../tline_simple/dataset_nocorpus_whitecolor/"+f)
        im = np.asarray(im)
        h,w,c = im.shape
        hnew = 32
        wnew = int( 1.0* hnew/h * w )
        t = cv2.resize( im,(wnew,hnew) )
        #for c in range(3):
        data[i,:,:wnew,:] = im[:,:,:]/255.0
        lengths.append(wnew/16)
        cnt = 0
        for it in labels[1:]:
            #if it=='0':
            #    break
            idx.append([i,cnt])
            cnt+=1
            vals.append( (int(it)-1)%ab_size )
    #print lengths
    return data,idx,vals,lengths

def rnn_layers(d,seql):
    seq_len=len(d)
    #seq_len,_,_ = d.get_shape()
    bs,fea_dim = d[0].get_shape().as_list()
    lstmcell1 = tf.contrib.rnn.BasicLSTMCell(fea_dim/2,state_is_tuple=True)
    lstmcell2 = tf.contrib.rnn.BasicLSTMCell(fea_dim/2,state_is_tuple=True)
    #cell = tf.contrib.rnn.MultiRNNCell([lstmcell1,lstmcell2],state_is_tuple=True)
    initstate = lstmcell1.zero_state(bs,tf.float32)
    out,fstate = tf.nn.bidirectional_dynamic_rnn(lstmcell1,lstmcell2,tf.stack(d),seql,initial_state_fw=initstate,initial_state_bw=initstate,time_major=True)
    #for i in range(seq_len):
    #    out,state = cell(d[i],state)
    return [tf.squeeze(t,[0]) for t in tf.split(tf.concat(out,-1),[1,]*seq_len)]

if __name__=='__main__':
    ab_size = 3851
    bs = 32
    imsize = [32,512,3]
    im = tf.placeholder(tf.float32,shape=[bs,]+imsize)
 
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = [bs,20]#tf.placeholder(tf.int64)
    gt_label = tf.SparseTensor(targetIxs, targetVals, targetShape)

    #with tf.variable_scope("cnn"):
    fea = getcnnfeature(im)
    n,h,w,c = fea.get_shape().as_list()
    fea = tf.transpose(fea,[2,0,1,3])
    fea = tf.reshape(fea,[w,n,h*c])
    feas = [tf.squeeze(t,[0]) for t in tf.split(fea,[1,]*w)]

    #feass = []
    #for i in range(len(feas)-1):
    #    tt = tf.concat([feas[i],feas[i+1]],axis=1)
    #    feass.append(tt)
    
    W = tf.get_variable("logit_weights", shape=[h*c,ab_size],initializer=tf.contrib.layers.xavier_initializer())
    B = tf.get_variable("logit_bias", shape=[ab_size],initializer=tf.contrib.layers.xavier_initializer())
    
    seqLengths = tf.placeholder(tf.int32)
    logits = [tf.matmul(t,W)+B for t in rnn_layers(feas,seqLengths)]
    print len(logits)
    #logits3d = tf.transpose(logits3d,[1,0,2])
    logits3d = tf.stack(logits)

    predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])   
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, gt_label, normalize=False)) / tf.to_float(tf.size(gt_label.values))

    loss = tf.reduce_mean( tf.nn.ctc_loss(gt_label,logits3d,seqLengths) )
    opt = tf.train.AdamOptimizer(1e-6).minimize(loss)

    trainlst = []
    for line in open("datalist/train_random_whitecolor.lst"):
        trainlst.append(line)
    testlst = []
    for line in open("datalist/test_random_whitecolor.lst"):
        testlst.append(line)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #for v in tf.trainable_variables():
        #    print v.name
        
        saver.restore(sess,tf.train.latest_checkpoint('ckpt_withrnn/whitecolor/'))
        """
        import pickle
        with open("datas.pickle",'rb') as f:
            datas = pickle.load(f)
        for name in datas:
            n=name.split(':')[0]
            print n
            if n.split('_')[0]=='logits':
                continue
            if n[:3] == 'rnn':
                continue
            with tf.variable_scope("",reuse=True):
                sess.run(tf.get_variable(n).assign(datas[name]))
        """
        for e in range(15):
            for i in range(len(trainlst)/bs):
                d,idx,vals,seqlength = getdata(trainlst,bs,i*bs,ab_size,imsize)
                loss_,_ = sess.run([loss,opt],{im:d,targetIxs:idx,targetVals:vals,seqLengths:seqlength})
                print "epoch:",e,"iter:",i,"loss:",loss_
                if i%100 == 0:
                    accuracy = 0
                    for j in range(len(testlst)/bs):
                        if random.randint(0,50) !=1:
                            continue
                        d,idx,vals,seqlength = getdata(testlst,bs,j*bs,ab_size,imsize)
                        acc = 1-sess.run(errorRate,{im:d,targetIxs:idx,targetVals:vals,seqLengths:seqlength})
                        accuracy += acc 
                        print "test iter:",j,"accuracy:",acc
                        break
                    #accuracy /= len(testlst)/bs
                    print "accuracy:",accuracy
                    saver.save(sess,"ckpt_withrnn/whitecolor/model.ckpt",e*len(trainlst)+i)
        saver.save(sess,"ckpt_withrnn/whitecolor/final.ckpt")
