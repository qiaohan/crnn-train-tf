# this project is a set for crnn's training and testing
the whole project contains scripts for :
* cnn pretrain
* train crnn without rnn
* train crnn with rnn(eg. one layer bidirection lstm)

to set up the project, you should pretrain the cnn first, and export the cnn's weights to initialize the crnn network

## train the cnn
	run the train code: python trainsinglechar.py

## export the weights: see the script test_crnn.py 
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

