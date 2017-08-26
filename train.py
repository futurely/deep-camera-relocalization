import math
import os

print 'from'
from posenet import GoogLeNet as PoseNet
from tensorflow.python.training import training_util
from tqdm import tqdm

print 'tensorflow'
import tensorflow as tf
import numpy as np

print 'from local'
from data import get_data, gen_data_batch
from net_builder import build_posenet, add_pose_loss

print 'global'
batch_size = 48
max_iterations = 30000
save_interval = 1000
validation_interval = 1000
# Set this path to your data_file data_dir
data_dir = '/home/user/Datasets/camera_relocalization/KingsCollege'
train_data_file = 'dataset_train.txt'
test_data_file = 'dataset_train.txt'
model_path = '/home/user/Datasets/tensorflow/models/mobilenet/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
restore_global_step = False
debug = False


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
  train_data_path = os.path.join(data_dir, train_data_file)
  train_data_source = get_data(train_data_path, data_dir)
  test_data_path = os.path.join(data_dir, test_data_file)
  test_data_source = get_data(test_data_path, data_dir)

  print 'build_posenet'
  net = build_posenet(images, 'mobilenet')
  #  net = PoseNet({'data': images})

  loss = add_pose_loss(net, poses_x, poses_q)
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
  p3_x = net['cls3_fc_pose_xyz']
  p3_q = net['cls3_fc_pose_wpqr']

  variables_to_restore = tf.trainable_variables()
  if restore_global_step:
    variables_to_restore.append(global_step)
  print 'variables_to_restore', variables_to_restore
  variables_to_save = tf.all_variables()

  restorer = tf.train.Saver(variables_to_restore)
  saver = tf.train.Saver(variables_to_save)

  checkpoint = tf.train.latest_checkpoint('.')
  if checkpoint is None:
    checkpoint = model_path
  output_file = 'PoseNet.ckpt'
  results = np.zeros((len(test_data_source.images), 2))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Load the data
    sess.run(init)
    # Restore model weights from previously saved model
    try:
      restorer.restore(sess, model_path)
    except:
      variables_to_restore = [
          x for x in tf.trainable_variables() if should_load(x.name)
      ]
      if restore_global_step:
        variables_to_restore.append(global_step)
      restorer = tf.train.Saver(variables_to_restore)
      restorer.restore(sess, checkpoint)
    print('Model restored from file: %s' % checkpoint)

    train_data_batch_generator = gen_data_batch(train_data_source, batch_size)
    test_data_batch_generator = gen_data_batch(test_data_source, batch_size)
    for i in range(max_iterations):
      np_images, np_poses_x, np_poses_q = next(train_data_batch_generator)
      feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

      sess.run(opt, feed_dict=feed)
      np_loss = sess.run(loss, feed_dict=feed)
      if i % 20 == 0:
        print('Iteration: ' + str(i) + '\n\t' + 'Loss is: ' + str(np_loss))
      if i > 0 and i % save_interval == 0:
        saver.save(sess, output_file, global_step=global_step)
        print('Intermediate file saved at: ' + output_file)
      if i > 0 and i % validation_interval == 0:
        print 'Validating'
        for j in tqdm(range(len(test_data_source.images))):
          np_image, np_poses_x, np_poses_q = next(test_data_batch_generator)
          feed = {images: np_image}

          pose_q = np.asarray(test_data_source.poses[j][3:7])
          pose_x = np.asarray(test_data_source.poses[j][0:3])
          predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)

          pose_q = np.squeeze(pose_q)
          pose_x = np.squeeze(pose_x)
          predicted_q = np.squeeze(predicted_q)
          predicted_x = np.squeeze(predicted_x)

          #Compute Individual Sample Error
          q1 = pose_q / np.linalg.norm(pose_q)
          q2 = predicted_q / np.linalg.norm(predicted_q)
          d = abs(np.sum(np.multiply(q1, q2)))
          theta = 2 * np.arccos(d) * 180 / math.pi
          error_x = np.linalg.norm(pose_x - predicted_x)
          results[j, :] = [error_x, theta]
          print 'Validation iter:  ', j, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

    saver.save(sess, output_file)
    print('Intermediate file saved at: ' + output_file)
    median_result = np.median(results, axis=0)
    print 'Median error ', median_result[0], 'm  and ', median_result[
        1], 'degrees.'


if __name__ == '__main__':
  main()
