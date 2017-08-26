# Import the converted model's class
import os

print 'from'
from posenet import GoogLeNet as PoseNet
from tensorflow.python.training import training_util

print 'tensorflow'
import tensorflow as tf

print 'from local'
from data import get_data, gen_data_batch
from net_builder import build_posenet, add_pose_loss

print 'global'
batch_size = 48
max_iterations = 30000
save_interval = 1000
# Set this path to your data_file data_dir
data_dir = '/home/user/Datasets/camera_relocalization/KingsCollege'
train_data_file = 'dataset_train.txt'
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
    for i in range(max_iterations):
      np_images, np_poses_x, np_poses_q = next(train_data_batch_generator)
      feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

      sess.run(opt, feed_dict=feed)
      np_loss = sess.run(loss, feed_dict=feed)
      if i % 20 == 0:
        print('iteration: ' + str(i) + '\n\t' + 'Loss is: ' + str(np_loss))
      if i > 0 and i % save_interval == 0:
        saver.save(sess, output_file, global_step=global_step)
        print('Intermediate file saved at: ' + output_file)
    saver.save(sess, output_file)
    print('Intermediate file saved at: ' + output_file)


if __name__ == '__main__':
  main()
