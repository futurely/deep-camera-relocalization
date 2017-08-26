# Import the converted model's class
import os

print 'from'
from posenet import GoogLeNet as PoseNet
from tensorflow.python.training import training_util
from tqdm import tqdm

print 'import'
import cv2
import numpy as np
import random
print 'tensorflow'
import tensorflow as tf

print 'from local'
from net_builder import build_posenet

print 'global'
batch_size = 48
max_iterations = 30000
# Set this path to your data_file data_dir
data_dir = '/home/user/Datasets/camera_relocalization/KingsCollege'
data_file = 'dataset_train.txt'
model_path = '/home/user/Datasets/tensorflow/models/mobilenet/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
restore_global_step = False
debug = False


class datasource(object):
  def __init__(self, images, poses):
    self.images = images
    self.poses = poses


def centeredCrop(img, output_side_length):
  height, width, depth = img.shape
  new_height = output_side_length
  new_width = output_side_length
  if height > width:
    new_height = output_side_length * height / width
  else:
    new_width = output_side_length * width / height
  height_offset = (new_height - output_side_length) / 2
  width_offset = (new_width - output_side_length) / 2
  cropped_img = img[height_offset:height_offset + output_side_length,
                    width_offset:width_offset + output_side_length]
  return cropped_img


def preprocess(images):
  images_out = []  #final result
  #Resize and crop and compute mean!
  images_cropped = []
  for i in tqdm(range(len(images))):
    print 'images[i]', i, images[i]
    X = cv2.imread(images[i])
    print 'image size', X.shape
    X = cv2.resize(X, (455, 256))
    X = centeredCrop(X, 224)
    images_cropped.append(X)
  #compute images mean
  N = 0
  mean = np.zeros((1, 3, 224, 224))
  for X in tqdm(images_cropped):
    mean[0][0] += X[:, :, 0]
    mean[0][1] += X[:, :, 1]
    mean[0][2] += X[:, :, 2]
    N += 1
  mean[0] /= N
  #Subtract mean from all images
  for X in tqdm(images_cropped):
    X = np.transpose(X, (2, 0, 1))
    X = X - mean
    X = np.squeeze(X)
    X = np.transpose(X, (1, 2, 0))
    images_out.append(X)
  return images_out


def get_data():
  poses = []
  images = []

  data_path = os.path.join(data_dir, data_file)
  with open(data_path) as f:
    next(f)  # skip the 3 header lines
    next(f)
    next(f)
    for line in f:
      fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
      p0 = float(p0)
      p1 = float(p1)
      p2 = float(p2)
      p3 = float(p3)
      p4 = float(p4)
      p5 = float(p5)
      p6 = float(p6)
      poses.append((p0, p1, p2, p3, p4, p5, p6))
      images.append(os.path.join(data_dir, fname))
      if debug and len(images) >= batch_size:
        break
  images = preprocess(images)
  return datasource(images, poses)


def gen_data(source):
  while True:
    indices = range(len(source.images))
    random.shuffle(indices)
    for i in indices:
      image = source.images[i]
      pose_x = source.poses[i][0:3]
      pose_q = source.poses[i][3:7]
      yield image, pose_x, pose_q


def gen_data_batch(source):
  data_gen = gen_data(source)
  while True:
    image_batch = []
    pose_x_batch = []
    pose_q_batch = []
    for _ in range(batch_size):
      image, pose_x, pose_q = next(data_gen)
      image_batch.append(image)
      pose_x_batch.append(pose_x)
      pose_q_batch.append(pose_q)
    yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)


def should_load(name):
  if name.startswith('cls') and name.find('_fc_pose_') != -1:
    return False
  if name.find('Logits') != -1 or name.find('Predictions') != -1:
    return False
  return True


def main():
  images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
  poses_x = tf.placeholder(tf.float32, [batch_size, 3])
  poses_q = tf.placeholder(tf.float32, [batch_size, 4])
  print 'get_data'
  datasource = get_data()

  print 'build_posenet'
  net = build_posenet(images, 'mobilenet')
  #  net = PoseNet({'data': images})

  loss = None
  try:
    p1_x = net['cls1_fc_pose_xyz']
    p1_q = net['cls1_fc_pose_wpqr']
    l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
    l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
    if loss is None:
      loss = l1_x + l1_q
    else:
      loss += l1_x + l1_q
  except:
    pass

  try:
    p2_x = net['cls2_fc_pose_xyz']
    p2_q = net['cls2_fc_pose_wpqr']
    l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
    l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
    if loss is None:
      loss = l2_x + l2_q
    else:
      loss += l2_x + l2_q
  except:
    pass

  try:
    p3_x = net['cls3_fc_pose_xyz']
    p3_q = net['cls3_fc_pose_wpqr']
    l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 0.3
    l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 150
    if loss is None:
      loss = l3_x + l3_q
    else:
      loss += l3_x + l3_q
  except:
    pass

  print 'loss', loss

  global_step = training_util.create_global_step()
  opt = tf.train.AdamOptimizer(
      learning_rate=0.0001,
      beta1=0.9,
      beta2=0.999,
      epsilon=0.00000001,
      use_locking=False,
      name='Adam').minimize(
          loss, global_step=global_step)

  # Set GPU options
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)

  init = tf.global_variables_initializer()
  variables_to_restore = tf.trainable_variables()
  if restore_global_step:
    variables_to_restore.append(global_step)
  print 'variables_to_restore', variables_to_restore
  #  variables_to_restore.append()
  saver = tf.train.Saver(variables_to_restore)
  output_file = 'PoseNet.ckpt'

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Load the data
    sess.run(init)
    # Restore model weights from previously saved model
    try:
      saver.restore(sess, model_path)
    except:
      variables_to_restore = [
          x for x in tf.trainable_variables() if should_load(x.name)
      ]
      if restore_global_step:
        variables_to_restore.append(global_step)
      saver = tf.train.Saver(variables_to_restore)
      saver.restore(sess, model_path)
    print('Model restored from file: %s' % model_path)

    data_gen = gen_data_batch(datasource)
    for i in range(max_iterations):
      np_images, np_poses_x, np_poses_q = next(data_gen)
      feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

      sess.run(opt, feed_dict=feed)
      np_loss = sess.run(loss, feed_dict=feed)
      if i % 20 == 0:
        print('iteration: ' + str(i) + '\n\t' + 'Loss is: ' + str(np_loss))
      if i > 0 and i % 1000 == 0:
        saver.save(sess, output_file, global_step=global_step)
        print('Intermediate file saved at: ' + output_file)
    saver.save(sess, output_file)
    print('Intermediate file saved at: ' + output_file)


if __name__ == '__main__':
  main()
