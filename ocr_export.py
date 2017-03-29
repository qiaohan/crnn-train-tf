# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7
"""Export inception model given existing training checkpoints.

The model is exported with proper signatures that can be loaded by standard
tensorflow_model_server.
"""

from __future__ import print_function

import os.path

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
#from inception import inception_model
from cnn import getcnnfeature,getcnnlogit

tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt_withoutrnn/colorcolor',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('export_dir', '/tmp/ocr_export',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('height', 32,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS


NUM_CLASSES = 1000
NUM_TOP_CLASSES = 5

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')

def inference(im):
    ab_size = 3851
    #bs = 1
    #imsize = [32,1024,3]
    #im = tf.placeholder(tf.float32,shape=[bs,]+imsize)
 
    #targetIxs = tf.placeholder(tf.int64)
    #targetVals = tf.placeholder(tf.int32)
    #targetShape = [bs,20]#tf.placeholder(tf.int64)
    #gt_label = tf.SparseTensor(targetIxs, targetVals, targetShape)

    fea = getcnnfeature(im)
    shape = tf.shape(fea)
    n = 1 #shape[0]
    h = 2 #shape[1]
    w = 1024/16 
    #w = shape[2]
    c = 512 #shape[3]
    fea = tf.transpose(fea,[2,0,1,3])
    fea = tf.reshape(fea,[w,n,h*c])
    print(w)
    feas = [tf.squeeze(t,[0]) for t in tf.split(fea,w)]
    print(len(feas))
    feass = []
    for i in range(len(feas)-1):
        tt = tf.concat([feas[i],feas[i+1]],axis=1)
        feass.append(tt)
    W = tf.get_variable("logit_weights", shape=[h*h*c,ab_size],initializer=tf.contrib.layers.xavier_initializer())
    B = tf.get_variable("logit_bias", shape=[ab_size],initializer=tf.contrib.layers.xavier_initializer())
    
    logits = [tf.matmul(t,W)+B for t in feass]
    #print len(logits)
    logits3d = tf.stack(logits)
    print(logits3d.get_shape().as_list())
    #logits3d = tf.transpose(logits3d,[1,0,2])
    #logits3d = tf.multiply(fea,W) + B

    seqLengths = [w-1] #tf.placeholder(tf.int32)
    predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d, seqLengths,merge_repeated = False)[0][0])
    return predictions.values

def export():
  '''
  # Create index->synset mapping
  synsets = []
  with open(SYNSET_FILE) as f:
    synsets = f.read().splitlines()
  # Create synset->metadata mapping
  texts = {}
  with open(METADATA_FILE) as f:
    for line in f.read().splitlines():
      parts = line.split('\t')
      assert len(parts) == 2
      texts[parts[0]] = parts[1]
  '''
  with tf.Graph().as_default():
    # Build inference model.
    # Please refer to Tensorflow inception model for details.

    # Input transformation.
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example['image/encoded']
    #images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
    images = preprocess_image(jpegs[0])
    #(serialized_tf_example)
    # Run inference.
    ss = inference(images)

    # Transform output to topK result.
    #values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)

    # Create a constant string Tensor where the i'th element is
    # the human readable class description for the i'th index.
    # Note that the 0th index is an unused background class
    # (see inception model definition code).
    '''
    class_descriptions = ['unused background']
    for s in synsets:
      class_descriptions.append(texts[s])
    class_tensor = tf.constant(class_descriptions)

    classes = tf.contrib.lookup.index_to_string(tf.to_int64(indices),
                                                mapping=class_tensor)

    '''
    # Restore variables from training checkpoint.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    inception_model.MOVING_AVERAGE_DECAY)
    #variables_to_restore = variable_averages.variables_to_restore()
    #saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        params = []
        names = []
        datas = {}
        import numpy as np
        for v in tf.trainable_variables():
            vd = v.eval(sess)
            print(v.name,vd.shape)
            vd.tofile("bin/"+v.name.split(':')[0]+".bin")
            #params.append( vd.copy() )
            #names.append( v.name )
            #datas[v.name] = vd.copy()
        return
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return

      # Export inference model.
      init_op = tf.group(tf.initialize_all_tables(), name='init_op')
      classification_signature = exporter.classification_signature(
          input_tensor=serialized_tf_example,
          classes_tensor=tf.constant(0),
          scores_tensor=tf.constant(0))
      named_graph_signature = {
          'inputs': exporter.generic_signature({'images': jpegs}),
          'outputs': exporter.generic_signature({
              'sentence': ss
              #'scores': values
          })}
      model_exporter = exporter.Exporter(saver)
      model_exporter.init(
          init_op=init_op,
          default_graph_signature=classification_signature,
          named_graph_signatures=named_graph_signature)
      model_exporter.export(FLAGS.export_dir, tf.constant(global_step), sess)
      print('Successfully exported model to %s' % FLAGS.export_dir)


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  shape = tf.shape(image)
  h = shape[0]
  w = shape[1]
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  #image = tf.image.central_crop(image, central_fraction=0.875)
  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  newh = tf.constant(FLAGS.height)
  #neww = tf.to_int32( newh * tf.div(h,w) )
  neww = tf.constant(1024)
  image = tf.image.resize_bilinear(image,
                                   [newh, neww],
                                   align_corners=False)
  #image = tf.squeeze(image, [0])
  # Finally, rescale to [-1,1] instead of [0, 1)
  #image = tf.subtract(image, 0.5)
  #image = tf.multiply(image, 2.0)
  return image


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
