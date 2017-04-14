import tensorflow as tf
import numpy as np
import cv2
from cnn import getcnnfeature,getcnnlogit

def getdata(datalst,batchsize,start,ab_size,imsize):
    label = np.zeros([batchsize,ab_size])
    data = np.zeros([batchsize,]+imsize)
    lengths = []
    idx = []
    vals = []
    for i in range(batchsize):
        labels = datalst[i+start].split()
        f = labels[0]
        im = cv2.imread("../../tline_simple/dataset_nocorpus/"+f)
        h,w,c = im.shape
        hnew = 32
        wnew = int( 1.0* hnew/h * w )
        t = cv2.resize( im,(wnew,hnew) )
        #for c in range(3):
        data[i,:,:wnew,:] = im[:,:,:]/255.0
        lengths.append(wnew/16)
        cnt = 0
        for it in labels[1:]:
            if it=='0':
                break
            idx.append([i,cnt])
            cnt+=1
            vals.append( (int(it)-1)%ab_size )
    #print lengths
    return data,idx,vals,lengths

def rnn_layers(d):
    seq_len=len(d)
    #seq_len,_,_ = d.get_shape()
    bs,fea_dim = d[0].get_shape().as_list()
    lstmcell1 = tf.contrib.rnn.BasicLSTMCell(fea_dim,state_is_tuple=True)
    lstmcell2 = tf.contrib.rnn.BasicLSTMCell(fea_dim,state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstmcell1,lstmcell2],state_is_tuple=True)
    initstate = cell.zero_state(bs,tf.float32)
    out,fstate = tf.nn.dynamic_rnn(cell,tf.stack(d),initial_state=initstate,time_major=True)
    #for i in range(seq_len):
    #    out,state = cell(d[i],state)
    return [tf.squeeze(t,[0]) for t in tf.split(out,[1,]*seq_len)]

word_dict=[]
for w in open("all_class_random.txt"):
    word_dict.append(w.strip().decode('utf-8'))
if __name__=='__main__':
    ab_size = 3817
    bs = 1
    imsize = [32,1024,3]
    im = tf.placeholder(tf.float32,shape=[bs,]+imsize)
 
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = [bs,20]#tf.placeholder(tf.int64)
    gt_label = tf.SparseTensor(targetIxs, targetVals, targetShape)

    fea = getcnnfeature(im)
    n,h,w,c = fea.get_shape().as_list()
    fea = tf.transpose(fea,[2,0,1,3])
    fea = tf.reshape(fea,[w,n,h*c])
    feas = [tf.squeeze(t,[0]) for t in tf.split(fea,[1,]*w)]

    #feass = []
    #for i in range(len(feas)-1):
    #    tt = tf.concat([feas[i],feas[i+1]],axis=1)
    #    feass.append(tt)
    
    W = tf.get_variable("logits_weights", shape=[h*c,ab_size],initializer=tf.contrib.layers.xavier_initializer())
    B = tf.get_variable("logits_bias", shape=[ab_size],initializer=tf.contrib.layers.xavier_initializer())
    
    logits = [tf.matmul(t,W)+B for t in rnn_layers(feas)]
    print len(logits)
    #logits3d = tf.transpose(logits3d,[1,0,2])
    logits3d = tf.stack(logits)

    seqLengths = tf.placeholder(tf.int32)
    predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])   
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, gt_label, normalize=False)) / tf.to_float(tf.size(gt_label.values))

    loss = tf.reduce_mean( tf.nn.ctc_loss(gt_label,logits3d,seqLengths) )
    opt = tf.train.AdamOptimizer(1e-5).minimize(loss)

    trainlst = []
    for line in open("datalist/train_random.lst"):
        trainlst.append(line)
    testlst = []
    for line in open("datalist/test_random.lst"):
        testlst.append(line)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        try:
            saver.restore(sess,tf.train.latest_checkpoint('ckpt_withrnn/'))
            #saver.restore(sess,tf.train.latest_checkpoint('ckpt32/'))
            params = []
            names = []
            datas = {}
            for v in tf.trainable_variables():
                vd = v.eval(sess)
                print v.name,vd.shape
                params.append( vd.copy() )
                names.append( v.name )
                datas[v.name] = vd.copy()
            import pickle
            with open("datas.pickle","wb") as f:
                pickle.dump(datas,f)
        except Exception,err:
            print err
        '''
        froot = "tline_simple/"
        import os
        for f in os.listdir(froot):
            img = cv2.imread("test1.jpg")
            #img = cv2.imread(froot+f)
            h,w,c = img.shape
            hnew = 32
            wnew = int( 1.0* hnew/h * w )
            img = cv2.resize( img,(wnew,hnew) )
            d = np.zeros([bs,]+imsize)
            d[0,:,:wnew,:] = img[:,:,:]/255.0
            pre = sess.run(predictions,{im:d,seqLengths:[wnew/16]})
            print "file:",f,"pre:",''.join([word_dict[t] for t in pre.values])#,"gt:",''.join([word_dict[t] for t in vals])
        '''
