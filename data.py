import os

from tqdm import tqdm
import cv2
import numpy as np
import random


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


def get_data(data_path, image_dir, max_num_images=-1):
  poses = []
  images = []

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
      images.append(os.path.join(image_dir, fname))
      if max_num_images > 0 and len(images) >= max_num_images:
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


def gen_data_batch(source, batch_size):
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
