import tensorflow as tf
import numpy as np
import cv2
from cnn2 import getcnnfeature,getcnnlogit
from PIL import Image

word_dict=[]
for w in open("word_dict.txt"):
#for w in open("all_class_random.txt"):
    word_dict.append(w.split()[1].strip().decode('utf-8'))
def getdata(datalst,batchsize,start,ab_size,imsize):
    data = np.zeros([batchsize,]+imsize)
    lengths = []
    idx = []
    vals = []
    for i in range(batchsize):
        labels = datalst[i+start].split()
        f = labels[0]
        #print f
        im = Image.open("../../tline_simple/dataset_nocorpus/"+f)
        im = np.asarray(im)
        s = random.randint(0,5)
        e = random.randint(0,2)
        im = im[s:e,:,:]
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

if __name__=='__main__':
    ab_size = 3851
    bs = 1
    imsize = [32,32,3]
    #imsize = [32,1024+512,3]
    im = tf.placeholder(tf.float32,shape=[bs,]+imsize)
 
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = [bs,20]#tf.placeholder(tf.int64)
    gt_label = tf.SparseTensor(targetIxs, targetVals, targetShape)

    fea,conv1 = getcnnfeature(im)
    n,h,w,c = fea.get_shape().as_list()
    
    feashape = fea.get_shape().as_list()
    print feashape
    feadim = feashape[-1]*feashape[-2]*feashape[-3]
    ffea = tf.reshape(fea,[feashape[0],-1])

    
    fea = tf.transpose(fea,[2,0,1,3])
    fea = tf.reshape(fea,[w,n,h*c])
    feas = [tf.squeeze(t,[0]) for t in tf.split(fea,[1,]*w)]

    feass = []
    for i in range(len(feas)-1):
        tt = tf.concat([feas[i],feas[i+1]],axis=1)
        feass.append(tt)
    W = tf.get_variable("logit_weights", shape=[h*h*c,ab_size],initializer=tf.contrib.layers.xavier_initializer())
    B = tf.get_variable("logit_bias", shape=[ab_size],initializer=tf.contrib.layers.xavier_initializer())
    
    logits = [tf.matmul(t,W)+B for t in feass]
    logits3d = tf.stack(logits)
    print(logits3d.get_shape().as_list())
    #logits3d = tf.transpose(logits3d,[1,0,2])
    #logits3d = tf.multiply(fea,W) + B

    seqLengths = tf.placeholder(tf.int32)
    predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d, seqLengths,merge_repeated = False)[0][0])   
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, gt_label, normalize=False)) / tf.to_float(tf.size(gt_label.values))

    loss = tf.reduce_mean( tf.nn.ctc_loss(gt_label,logits3d,seqLengths) )
    opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

    tttt = tf.matmul(ffea,W)+B
    
    trainlst = []
    for line in open("datalist/train_random.lst"):
        trainlst.append(line)
    testlst = []
    for line in open("datalist/test_random.lst"):
        testlst.append(line)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,tf.train.latest_checkpoint('ckpt_withoutrnn/colorcolor/'))
        '''
        import numpy as np
        shapes = {}
        shapes["conv1_weights"] = [3,3,3,64]
        shapes["conv1_bias"] = [64]
        shapes["conv2_weights"] = [3,3,64,128]
        shapes["conv2_bias"] = [128]
        shapes["conv3_weights"] = [3,3,128,256]
        shapes["conv3_bias"] = [256]
        shapes["conv4_weights"] = [3,3,256,256]
        shapes["conv4_bias"] = [256]
        shapes["conv5_weights"] = [3,3,256,512]
        shapes["conv5_bias"] = [512]
        shapes["conv6_weights"] = [3,3,512,512]
        shapes["conv6_bias"] = [512]
        shapes["conv7_weights"] = [3,3,512,512]
        shapes["conv7_bias"] = [512]
        shapes["logit_weights"] = [2048,3851]
        shapes["logit_bias"] = [3851]
        shapes["bn5_offset"] = [512]
        shapes["bn5_scale"] = [512]
        shapes["bn6_offset"] = [512]
        shapes["bn6_scale"] = [512]
        for v in tf.trainable_variables():
            vd = np.fromfile("bin/"+v.name.split(':')[0]+".bin",np.float32)
            vd = vd.reshape(shapes[v.name.split(':')[0]])
            print(v.name,vd.shape)
            v.assign(vd) 
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
        with open("dataswithoutrnn.pickle","wb") as f:
            pickle.dump(datas,f)
        #np.savez('params.npz',name = names, param = params)
        #np.savez('name.npz',names)
        ''' 
        #froot = "tline_3/"
        #froot = "tline_simple/"
        #froot = "testtmp/"
        #froot="/data1/hzqiaohan/yidun/gen_ad_img/tline_simple/sample/"
        #froot="/home/hzqiaohan/OCR/e2e/OCR_Sample_Gen/sample/"
        froot = "/home/hzqiaohan/OCR/CTPN/wxt/white/"

        import os
        import codecs
        fff = codecs.open("res.txt","w","utf-8")
        for f in os.listdir(froot):
            img = Image.open(froot+f)
            img = np.asarray(img)
            h,w,c = img.shape
            hnew = 32 #- 1 - 2
            wnew = int( 1.0* hnew/h * w )
            print "new width:",wnew
            img = cv2.resize( img,(wnew,hnew) )
            d = np.ones([bs,]+imsize,np.float32)
            #d[0,2:31,:wnew,:] = img[:,:,:]/255.0
            #d[0,:,:wnew,:] = img[:,:,:]
            #dd = np.ones(imsize)*255
            #dd[:,:wnew,:] = img[:,:,:]
            #imm = Image.fromarray(np.uint8(dd))
            #imm.save(f)
            bb = sess.run(feass[0],{im:d,seqLengths:[wnew/16]})
            #aa = sess.run(conv1,{im:d,seqLengths:[wnew/16]})
            aa = sess.run(logits[0],{im:d,seqLengths:[wnew/16]})
            #print aa[0,0,:,0]
            #print aa[0,1,:,0]
            #print aa[0,1,:,0]
            #print [aa[:,:,:,i].var() for i in range(512)]
            for i in range(3851):
                print aa[0,i]
            #for i in range(2048):
            #    print bb[0][i]
            print aa.shape
        
            #conv1 = sess.run(tf.get,{im:d,seqLengths:[wnew/16-1]})
            break
            pre = sess.run(predictions,{im:d,seqLengths:[wnew/16-1]})
            #print pre.values
            print "file:",f,"pre:",''.join([word_dict[t] for t in pre.values])
            #,"gt:",''.join([word_dict[t] for t in vals])
            try:
                fff.write(f+" "+''.join(word_dict[t] for t in pre.values)+'\n')
            except Exception,e:
                print e
        fff.close()
