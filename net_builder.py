print '  fromt'
from slim.nets import mobilenet_v1 as mobilenet
print '  import'
import tensorflow as tf


def add_predictions(net, end_points):
  pose_xyz = tf.layers.dense(net, 3, name='cls3_fc_pose_xyz')
  end_points['cls3_fc_pose_xyz'] = pose_xyz
  pose_wpqr = tf.layers.dense(net, 4, name='cls3_fc_pose_wpqr')
  end_points['cls3_fc_pose_wpqr'] = pose_wpqr


def build_posenet(inputs, net_type):
  if net_type.startswith('mobilenet'):
    net = mobilenet.mobilenet_v1
    logits, end_points = net(inputs, num_classes=1001)
  if net_type.startswith('mobilenet'):
    net = end_points['AvgPool_1a']
  add_predictions(net, end_points)
  return end_points
