from slim.nets import mobilenet_v1 as mobilenet
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf


def add_predictions(net, end_points):
  pose_xyz = tf.layers.dense(
      net, 3, name='cls3_fc_pose_xyz', kernel_initializer=xavier_initializer())
  end_points['cls3_fc_pose_xyz'] = pose_xyz
  pose_wpqr = tf.layers.dense(
      net,
      4,
      name='cls3_fc_pose_wpqr',
      kernel_initializer=xavier_initializer())
  end_points['cls3_fc_pose_wpqr'] = pose_wpqr


def build_posenet(inputs, net_type):
  if net_type.startswith('mobilenet'):
    net = mobilenet.mobilenet_v1
    logits, end_points = net(inputs, num_classes=1001)
  if net_type.startswith('mobilenet'):
    net = end_points['AvgPool_1a']
  add_predictions(net, end_points)
  return end_points


def add_pose_loss(net, poses_x, poses_q):
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

  return loss
