import math
import os

import numpy as np
import tensorflow as tf

from data import get_data, gen_data_batch
from net_builder import build_posenet

batch_size = 1

# Set this path to your project directory
path = 'path_to_project/'
# Set this path to your dataset directory
data_dir = '/home/user/Datasets/camera_relocalization/KingsCollege'
data_file = 'dataset_test.txt'


def main():
  images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
  data_path = os.path.join(data_dir, data_file)
  test_data_source = get_data(data_path, data_dir)
  results = np.zeros((len(test_data_source.images), 2))

  net = build_posenet(images, 'mobilenet')

  p3_x = net['cls3_fc_pose_xyz']
  p3_q = net['cls3_fc_pose_wpqr']

  init = tf.global_variables_initializer()
  checkpoint = tf.train.latest_checkpoint('.')

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Load the data
    sess.run(init)
    saver.restore(sess, checkpoint)

    test_data_batch_generator = gen_data_batch(test_data_source, batch_size)
    for i in range(len(test_data_source.images)):
      np_image, np_poses_x, np_poses_q = next(test_data_batch_generator)
      feed = {images: np_image}

      pose_q = np.asarray(test_data_source.poses[i][3:7])
      pose_x = np.asarray(test_data_source.poses[i][0:3])
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
      results[i, :] = [error_x, theta]
      print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

  median_result = np.median(results, axis=0)
  print 'Median error ', median_result[0], 'm  and ', median_result[
      1], 'degrees.'


if __name__ == '__main__':
  main()
