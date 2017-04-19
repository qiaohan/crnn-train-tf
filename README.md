# this project is a set for crnn's training and testing
the whole project contains scripts for :
* cnn pretrain
* train/test crnn without rnn
* train/test crnn with rnn(eg. one layer bidirection lstm)

to set up the project, you should pretrain the cnn first, and export the cnn's weights to initialize the crnn network
the train/test .lst file has format as:
XXX.png 100 200 333 666
split by blank, and the first element is image file name, rest the char's index(file word_dict.txt record all the chars and their index number)

# dictionary
the file word_dict.txt records all the characters which can be recognized: 3754 chinese, 10 number, 52 english, and other 34 symbol

## train the cnn
	run the train code: python trainsinglechar.py

## export the cnn weights for crnn initialization: see the script test_crnn.py 
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
## export the crnn weights for cpp server (implimented with cudnn or tenserflow_servering): see ocr_export.py
	for cudnn impliment --------
				for v in tf.trainable_variables():
		            vd = v.eval(sess)
				    print(v.name,vd.shape)
				    vd.tofile("bin/"+v.name.split(':')[0]+".bin")
	for tensorflow_servering --------
				 see ocr_export.py or tensorflow website/api/deploy
## train/test the crnn without rnn
	train : python traintline.py
	test : python test_tline_without.py

## train/test the crnn with rnn
	train : python train_crnn.py
	test : python test_crnn.py
