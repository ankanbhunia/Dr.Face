import sys
sys.path.append('/content/FSA-Net')
import dlib
import face_detection
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from lib.FSANET_model import *
import numpy as np
from keras.layers import Average
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
mtcnn = MTCNN(keep_all=True, device='cuda')
from collections import defaultdict 
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm.notebook import tqdm
import imutils
predictor_path = '/content/FaceClust/shape_predictor_5_face_landmarks.dat' # Download from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
face_rec_model_path = '/content/FaceClust/dlib_face_recognition_resnet_model_v1.dat' # Download from http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
import dlib
sp = dlib.shape_predictor(predictor_path) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model
detector = dlib.cnn_face_detection_model_v1('/content/FaceClust/mmod_human_face_detector.dat')

stage_num = [3,3,3]
lambda_local = 1
lambda_d = 1
img_idx = 0
detected = '' #make this not local variable
time_detection = 0
time_network = 0
time_plot = 0
skip_frame = 1 # every 5 frame do 1 detection and network forward propagation
ad = 0.6

#Parameters
num_capsule = 3
dim_capsule = 16
routings = 2
stage_num = [3,3,3]
lambda_d = 1
num_classes = 3
image_size = 64
num_primcaps = 7*3
m_dim = 5
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

num_primcaps = 8*8*3
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

print('Loading models ...')

weight_file1 = '/content/FSA-Net/pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
model1.load_weights(weight_file1)
print('Finished loading model 1.')

weight_file2 = '/content/FSA-Net/pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
model2.load_weights(weight_file2)
print('Finished loading model 2.')

weight_file3 = '/content/FSA-Net/pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
model3.load_weights(weight_file3)
print('Finished loading model 3.')

inputs = Input(shape=(64,64,3))
x1 = model1(inputs) #1x1
x2 = model2(inputs) #var
x3 = model3(inputs) #w/o
avg_model = Average()([x1,x2,x3])

model = Model(inputs=inputs, outputs=avg_model)

detector = face_detection.build_detector(
  "RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from numba import jit
import os.path
import numpy as np
##import matplotlib.pyplot as plt
##import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

@jit
def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.objclass = 5# bbox[6]

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1], [trk.objclass])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

def EXTRACT_INFO(v_cap):

    DT = defaultdict(dict)
    
    tracker =  Sort()
    #v_cap = VideoFileClip(path)
    v_cap = v_cap.resize(height=480)
    b_size = 5

    data = {}
    data['BBOX'] =[]
    data['Frame_ID'] =[]
    data['ANGLE'] =[]
    data['TRACK_ID'] =[]
    data['BBOX_FEAT'] = []
    frames = []
    store_tmp_idx = []
    for idx, frame in tqdm(enumerate(v_cap.iter_frames()), total = v_cap.reader.nframes):

      frames.append(Image.fromarray(frame))
      store_tmp_idx.append(idx)

      if idx%b_size==0:

        #boxes, pr, landmarks = mtcnn.detect(frames,landmarks=True)
        boxes = detector.batched_detect(np.stack(frames,0))

        bbox_list = []

        for fr,bbs,fridx in zip(frames, boxes, store_tmp_idx):
          
          if bbs is not None:

            trackers = tracker.update(bbs)

            for b in trackers:
              
              bbox_fr_ = np.array(fr.crop(b[:4]))
              bbox_fr = cv2.resize(bbox_fr_, (64,64))
              bbox_fr = np.expand_dims(bbox_fr, axis=0)
              bbox_list.append(bbox_fr)

              if len(DT[int(b[4])])== 0:
                DT[int(b[4])]['IMG'] = []#bbox_fr_

              DT[int(b[4])]['IMG'].append( bbox_fr_)

              feats_ = facerec.compute_face_descriptor(np.array(fr), sp(np.array(fr), dlib.rectangle(int(b[0]),int(b[1]),int(b[2]),int(b[3]))))
            
              data['BBOX'].append(b[:4])
              data['Frame_ID'].append(fridx)
              data['TRACK_ID'].append(int(b[4]))
              data['BBOX_FEAT'].append(feats_)
            
        if len(bbox_list)>0: data['ANGLE'].extend(model.predict(np.concatenate(bbox_list, 0)))
        frames = []
        store_tmp_idx = []

    labels = data['TRACK_ID']
    

    for p in list(set(labels)):

      indx_p = np.where([labels == np.ones_like(labels)*p])[-1]
      
      if len(indx_p)>10:
        DT[p]['BBOX'] = np.array(data['BBOX'])[indx_p]
        DT[p]['Frame_ID'] = np.array(data['Frame_ID'])[indx_p]
        DT[p]['BBOX_FEAT'] = np.array(data['BBOX_FEAT'])[indx_p]
        DT[p]['ANGLE'] = np.array(data['ANGLE'] )[indx_p]
        
    return DT