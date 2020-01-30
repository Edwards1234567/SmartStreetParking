import fileinput
import os, re,smtplib, subprocess,sys
import requests
from flask import Flask, render_template, session, request, redirect, url_for,send_from_directory
from collections import OrderedDict
from urllib.parse import *
from datetime import datetime
from wtforms import Form, HiddenField
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shutil

app = Flask(__name__)


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import pylivecap

import matplotlib

import datetime

import json


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import db


cred = credentials.Certificate("../smartstreetparking-5038c-firebase-adminsdk-hn59r-34a21fc9da.json")
firebase_admin.initialize_app(cred)
fire_db= firestore.client()
doc_ref = fire_db.collection('spots')


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

#MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box'
#MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box'
MODEL_FILE = 'object_detection'+MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'object_detection/'+MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/'+'data', 'mscoco_label_map.pbtxt')


global number
number = 6
right_border = 1
left_border = 0
global used
used = 0
global ts
ts="1"
n = 1
img="image_"

global spots
spots = [0]*number

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
TEST_IMAGE_PATHS = []
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.JPG'.format(i)) for i in range(1, 10) ]
TEST_IMAGE_PATHS.append(os.path.join('test_images','image1.jpg'))
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


LINK="https://www.youtube.com/watch?v=vCDDYb_M2B4"
#LINK="https://www.twitch.tv/jchengli"
#LINK="https://www.twitch.tv/lunalalanana"




@app.route('/shoot', methods=['GET'])
def shoot():
    print('shoot')

    message = 'heyyy please take a shot'
    return render_template('shot.html', message=message)
    #return jsonify(message = message)

@app.route('/map', methods=['GET'])
def map():
    print('map')

    message = 'please see the map below'
    return render_template('map.html', message=message)
    #return jsonify(message=message)

@app.route('/api', methods=['GET'])
def api():
    print('api')

    #message = 'please see the map below'
    print(used)
    return render_template('Parking-luna.html',total_number=number,vacant_number=number-used, spots=spots)
    #return jsonify(total_number=number,vacant_number=number-used, spots=spots)


@app.route('/detect', methods=['GET'])
def detect():
    print('detect')
    #gp.downloadLastMedia(custom_filename='object_detection/test_images/image.jpg')
    message = 'please click detect to analysis the latest photo'
    image = os.path.join(img+ts+'.jpg')
    print(image)
    #image = os.path.join('object_detection/test_images/image1.jpg')
    return render_template('detect.html', message=message, image=image)
    #return jsonify(message=message, image=image)

@app.route('/shootphoto', methods=['POST'])
def shootphoto():
    print('shootphoto')
    # gp.take_photo()
    w = datetime.datetime.now()
    w = str(w)
    w = w.replace(" ","")
    w = w.replace(".","")
    w = w.replace(":","")
    global ts
    ts = w.replace("-","")
    pylivecap.safe_capture(LINK, "static/"+img+ts+".jpg")
    message = 'photo shot, please go next step'
    return render_template('shot.html', message=message)
    #return jsonify(message)

@app.route('/detectphoto', methods=['POST'])
def detectphoto():
    print('detecttphoto')
    # gp.take_photo()

    image_path = "static/"+img+ts+".jpg"
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    # print(output_dict['detection_classes'])
    # print(output_dict['detection_boxes'])

    box_to_display_str_map, box_to_color_map = display_classes(
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index)
    print("==================================")
    print(box_to_display_str_map)
    print("==================================")

    global spots
    spots = [0]*number


    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        print(str(ymin) + " " + str(ymax) + " " + str(xmin) + " " + str(xmax))
        center = (xmin + xmax)/2
        print(center)
        margin = (right_border-left_border)*1.0/number
        rest = center % margin
        pos = int((center - rest)/margin)
        print(pos)
        if spots[pos] == 0:
            spots[pos] = 1
    print("==================================")

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.savefig('static/'+img+ts+'ext.jpg')
    #image_np.save('static/image1ext.jpg')


    docs = doc_ref.stream()
    for doc in docs:
        dd = doc.to_dict()
        if spots[int(dd['spot_id'])] == 0:

            fields = {'status': 'vacant'}
            #         doc_ref.document(doc.id).update(fields)
            doc.reference.update(fields)
        else:
            fields = {'status': 'occupied'}
            #         doc_ref.document(doc.id).update(fields)
            doc.reference.update(fields)

    ret= {}
    ret['first'] = []
    for i in range(len(spots)):
        temp = {}
        temp['id'] = i
        temp['used'] = spots[i]
        ret['first'].append(temp)

    #ret = json.dumps(ret)
    print("test json")
    print(ret)
    image = img+ts+'ext.jpg'
    global used
    print(box_to_display_str_map)
    used = len(box_to_display_str_map)
    message = 'this is the latest photo:'
    #str(len(box_to_display_str_map))
    return render_template('detect.html', message=message,image=image, spots = spots)
    #return jsonify(message=message,image=image, spots = ret)

@app.route('/')
def hello_world():
    message = 'Welcome'
    return render_template('base.html', message=message)
    #return jsonify(message=message1)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


import collections


def display_classes(
        boxes,
        classes,
        scores,
        category_index,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        skip_scores=False,
        agnostic_mode=False,
        skip_labels=False):
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_display_str_list = []
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())

            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100 * scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                box_to_display_str_list.append(display_str)

                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)]

    return box_to_display_str_list, box_to_color_map



STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
if __name__ == '__main__':
    app.run(debug=type)

