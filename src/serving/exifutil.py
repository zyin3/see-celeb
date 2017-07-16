"""This script handles the skimage exif problem.
"""

from PIL import Image
import numpy as np
import tensorflow as tf
from align import detect_face
import facenet
from scipy import misc
import os
import random
import datetime

ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}

def open_oriented_im(im_path):
  im = Image.open(im_path)
  if hasattr(im, '_getexif'):
    exif = im._getexif()
    if exif is not None and 274 in exif:
      orientation = exif[274]
      im = apply_orientation(im, orientation)
  img = np.asarray(im).astype(np.float32) / 255.
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    img = np.tile(img, (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img


def apply_orientation(im, orientation):
  if orientation in ORIENTATIONS:
    for method in ORIENTATIONS[orientation]:
      im = im.transpose(method)
  return im


def crop_and_align(image_path, margin=44, image_size=182):
  with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

  minsize = 20  # minimum size of face
  threshold = [0.6, 0.7, 0.7]  # three steps's threshold
  factor = 0.709  # scale factor

  # print(image_path)
  # filename = os.path.splitext(os.path.split(image_path)[1])[0]
  output_filename = os.path.join('/tmp/demos_uploads',
                                 str(datetime.datetime.now()).replace(' ', '_') + '.png')

  if not os.path.exists(output_filename):
    try:
      img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
      errorMessage = '{}: {}'.format(image_path, e)
      print(errorMessage)
    else:
      print("img is read succesfully.")
      if img.ndim < 2:
        raise ValueError('Unable to align')
      if img.ndim == 2:
        img = facenet.to_rgb(img)
      img = img[:, :, 0:3]

      bounding_boxes, _ = detect_face.detect_face(
          img, minsize, pnet, rnet, onet, threshold, factor)
      nrof_faces = bounding_boxes.shape[0]
      if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
          bounding_box_size = (det[:, 2] - det[:, 0]) * (
              det[:, 3] - det[:, 1])
          img_center = img_size / 2
          offsets = np.vstack(
              [(det[:, 0] + det[:, 2]) / 2 - img_center[1],
               (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
          offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
          index = np.argmax(bounding_box_size - offset_dist_squared * 2.0
                           )  # some extra weight on the centering
          det = det[index, :]
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        scaled = misc.imresize(
            cropped, (image_size, image_size),
            interp='bilinear')
        misc.imsave(output_filename, scaled)
        return output_filename
      else:
        print('Unable to align')
