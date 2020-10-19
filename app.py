from importlib import reload  
import zipfile
import tqdm
from subprocess import getoutput
import imutils
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import flask
from flask import Flask, Response
import threading
import cv2
import time
import io
import os
import base64
import dash_daq as daq
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import youtube_dl
from shutil import copyfile
import shutil
import glob
import timeago, datetime
import psutil
from moviepy.editor import VideoFileClip, concatenate_videoclips
import sys
IN_COLAB = 'COLAB_GPU' in os.environ
if IN_COLAB: PYTHON_PATH  = 'Library/bin/python' 
else: PYTHON_PATH = 'python3.6'
import queue
import random
import string
from multiprocessing import Process, Queue
#global thread_list
thread_list = []
import subprocess
global subprocess_list
subprocess_list = []
from subprocess import Popen
import pickle
import signal
import dash_editor_components
from flask import request
import sys
sys.path.append('FaceClust')
import FaceClust.face_clustering as ffc
global cvt_id
cvt_id = None
global open_choose_box
open_choose_box = False
global npy_files
npy_files = [i for i in os.listdir('/tmp') if i.endswith('.npy')]
from multiprocessing import Process, Value,Manager
import GPUtil
global Progress_modal_is_open
Progress_modal_is_open = True
import sys  
sys.path.append('DeepFaceLab')
counter_children = 0
import numpy as np
from facelib import FaceType
global labelsdict
global run
global no_loop
no_loop = False
run = Value("i", 0)
manager = Manager()
global total_src_frames
global total_src_frames_paths
total_src_frames = 0
total_src_frames_paths = []
global src_face_list
src_face_list = []
global total_dst_frames
total_dst_frames = 0
global total_dst_frames_paths
total_dst_frames_paths = []
global dst_face_list
dst_face_list = []
import sys
sys.path.append('dash-player')
global convert_disabled
convert_disabled = False
import dash_player
import argparse
from random import *
if not os.path.isdir('/data'): os.mkdir('/data')
import atexit
if os.path.isfile('/tmp/running'): os.remove('/tmp/running')
if os.path.isfile('/tmp/processing'): os.remove('/tmp/processing')
if os.path.isfile('/tmp/ResourceExhaustedError'): os.remove('/tmp/ResourceExhaustedError')
if os.path.isfile('/tmp/converting'): os.remove('/tmp/converting')
if os.path.isfile('/tmp/start'): os.remove('/tmp/start')
if os.path.isdir('/tmp/cluster'): shutil.rmtree('/tmp/cluster')
for filename in glob.glob("assets/*.mp4"):os.remove(filename)
global show_mode
show_mode = 1
parser = argparse.ArgumentParser(description='dr.face Options')
parser.add_argument('drivepath', type=str, nargs='?',
                    help='Enter ngrok Authtoken from https://dashboard.ngrok.com/auth/your-authtoken ')
argss = parser.parse_args()
if argss.drivepath == None:
    argss.drivepath = 'drive/My Drive/'
drive_path = argss.drivepath
IN_COLAB_DRIVE = IN_COLAB and os.path.isdir(drive_path)
if os.path.isdir(drive_path):
  if not os.path.isdir(os.path.join(drive_path, 'Dr.Face')):os.mkdir(os.path.join(drive_path, 'Dr.Face'))
def kill_pythons():
    import psutil
    import os
    for proc in psutil.process_iter():
        pinfo = proc.as_dict(attrs=['pid', 'name'])
        procname = str(pinfo['name'])
        procpid = str(pinfo['pid'])
        if "python" in procname and procpid != str(os.getpid()):
            #print("Stopped Python Process ", proc)
            proc.kill()
def killall():
    
    os.system('fuser -k /dev/nvidia-uvm > /dev/null 2>&1')
    
          
killall()
def exit_handler():
    
    #print ('\nexiting program.... ')
    killall()
    killall()
    kill_pythons()
    
    #os.system('rm -r /tmp/*npy')
    for filename in glob.glob("/tmp/*npy"):os.remove(filename)
    for filename in glob.glob("assets/*.mp4"):os.remove(filename)
    #os.system('rm /tmp/model.txt')
    #print('\nCompleted.')
    
    if not IN_COLAB: os.system('fuser -k /usr/bin/python3.6')
    
    
if not IN_COLAB: atexit.register(exit_handler)
class merging_vars:
  def __init__(self, 
                #face_type = None,
                output_face_scale = 0,
                super_resolution_power = 0,
                mask_mode = 3,
                mode = 'overlay',
                erode_mask_modifier = 0,
                blur_mask_modifier = 0,
                color_transfer_mode = 0,
                masked_hist_match = True,
                hist_match_threshold = 255,
                motion_blur_power = 0,
                blursharpen_amount = 0,
                image_denoise_power = 0,
                bicubic_degrade_power = 0,
                sharpen_mode = 1,
                color_degrade_power = 0,
                horizontal_shear = 0,
                vertical_shear = 0,
                horizontal_shift = 0,
                vertical_shift = 0,
                show_mode = 1
                ):
    #self.face_type = face_type
    self.output_face_scale = output_face_scale
    self.super_resolution_power = super_resolution_power
    self.mask_mode = mask_mode
    self.mode = mode
    self.erode_mask_modifier = erode_mask_modifier
    self.blur_mask_modifier = blur_mask_modifier
    self.color_transfer_mode = color_transfer_mode
    self.masked_hist_match = masked_hist_match
    self.hist_match_threshold = hist_match_threshold
    self.motion_blur_power = motion_blur_power
    self.blursharpen_amount = blursharpen_amount
    self.image_denoise_power = image_denoise_power
    self.bicubic_degrade_power = bicubic_degrade_power
    self.sharpen_mode = sharpen_mode
    self.color_degrade_power = color_degrade_power
    self.horizontal_shear = horizontal_shear
    self.vertical_shear = vertical_shear
    self.horizontal_shift = horizontal_shift
    self.vertical_shift = vertical_shift
    self.show_mode = show_mode
#[os.remove(os.path.join('/tmp',i)) for i in os.listdir('/tmp') if i.endswith('.npy')]
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
with open('/tmp/log.txt', 'w') as f:
        f.close()
        
        
def run_cmd(cmd):
    p = subprocess.Popen("exec " + cmd, shell=True)
    
    
    with open('/tmp/log.txt', 'a') as f:
        f.write(str(p.pid)+ '\n')
        f.close()
        
    p.wait()
#if not os.path.isdir(datadir()+''): os.mkdir(datadir()+'')
#if not os.path.isdir(datadir()+'/data_dst'): os.mkdir(datadir()+'/data_dst')
#if not os.path.isdir(datadir()+'/data_src'): os.mkdir(datadir()+'/data_src')
#if not os.path.isdir(datadir()+'/model'): os.mkdir(datadir()+'/model')
#if not os.path.isdir(datadir()+'/data_dst'): os.mkdir(datadir()+'/data_dst')
if not os.path.isfile('/tmp/model.txt'):
#    convert_id = (''.join(map(choice,["bcdfghjklmnpqrstvwxz","aeiouy"]*3)))
            
    f = open('/tmp/model.txt','w+')
   # f.write(convert_id)
    f.close()
    
def datadir():
     f = open('/tmp/model.txt','r')
     n = f.read()
     return os.path.join('/data', n)
    
from inspect import currentframe, getframeinfo
class VideoCamera(object):
    def __init__(self):
        self.open = True
        self.fourcc = "VID"
        self.video = cv2.VideoCapture(0)
        # self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter('videos/Record/temp.mp4', -1, 20.0, (640,480))
        self.frame_counts = 1
    def __del__(self):
        self.video.release()
    def get_frame(self):
        
        try:
            self.success, self.image = self.video.read()
            ret, jpeg = cv2.imencode('.jpg', self.image)
            return jpeg.tobytes()
        except:
            pass
    
    def record(self):
        timer_start = time.time()
        timer_current = 0
        while(self.open==True):
            try:
                ret, video_frame = self.success, self.image
              
            except:
                break
            if (ret==True):
                    self.video_out.write(video_frame)
                    self.frame_counts += 1
                    time.sleep(1/10)
            else:
                break
    def stop(self):
        if self.open==True:
            self.open=False
            self.video_out.release()
            self.video.release()
            cv2.destroyAllWindows()
        else: 
            pass
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()
def gen(camera):
    while camera.open:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def get_sec2time(s):
    hours, rem = divmod(s, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    if hours == 0:
      if  minutes <10:
        if seconds <10:
          return '0'+str(minutes)+':0'+str(seconds)
        else:
          return '0'+str(minutes)+':'+str(seconds)
      else:
        return str(minutes)+':'+str(seconds)
    else:
      return str(hours)+':'+str(minutes)+':'+str(seconds)
def get_interval_func(start_time):
    hours, rem = divmod(time.time()-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    def sec(s):
      if s == 0:
        return ''
      elif s == 1:
        return str(1) + ' second '
      else:
        return str(s) + ' seconds'
    def min(s):
      if s == 0:
        return ''
      elif s == 1:
        return str(1) + ' minute '
      else:
        return str(s) + ' minutes '
    def hour(s):
      if s == 0:
        return ''
      elif s == 1:
        return str(1) + ' hour '
      else:
        return str(s) + ' hours '
    return str(hours)+':'+str(minutes)+':'+str(seconds)#hour(hours) + min(minutes) + sec(seconds)
class stopWatch:
  def __init__(self):
    pass
  def start(self):
    self.start_time = time.time() 
  def end(self):
    self.end_time = time.time()
  def get_interval(self):
    return get_sec2time(time.time()-self.start_time)
def Convert():
    f = open('/tmp/model.txt','r')
    convert_id = f.read()
    f.close()
        
      
    output_name =  convert_id + '.mp4'
    ###########print ('###############################' + output_name
    if not os.path.isdir(datadir()+'/data_dst/merged'): os.mkdir(datadir()+'/data_dst/merged')
    if not os.path.isdir(datadir()+'/data_dst/merged_mask'): os.mkdir(datadir()+'/data_dst/merged_mask')
    os.system('echo | '+PYTHON_PATH+' DeepFaceLab/main.py merge --input-dir '+datadir()+'/data_dst --output-dir '+datadir()+'/data_dst/merged --output-mask-dir '+datadir()+'/data_dst/merged_mask --aligned-dir '+datadir()+'/data_dst/aligned --model-dir '+datadir()+'/model --model SAEHD')
    os.system('echo | '+PYTHON_PATH+' DeepFaceLab/main.py videoed video-from-sequence --input-dir '+datadir()+'/data_dst/merged --output-file '+datadir()+'/'+output_name+' --reference-file '+datadir()+'/data_dst.mp4 --include-audio')
    #os.system('cp '+datadir()+'/'+output_name+' /data')
    
    
    for filename in glob.glob("/tmp/*npy"):os.remove(filename)
    # need to install xattr
    
    ###########print ('###############################' + 'convertion done')
    
def save_workspace_data():
    time.sleep(3600)
    while 1:
       
      
      f = open('/tmp/model.txt','r')
      convert_id = f.read()
      f.close()
      #print (convert_id)
      #print ('jjkdhsjksjkdkdkdkdkldkdkdkdlld#@@@@@@@@@@@@@@@@@' + convert_id)
      #print ('zip -r -q '+convert_id+'.zip '+datadir())
      os.system('zip -r -q '+convert_id+'.zip '+datadir()); 
      copyfile(convert_id+'.zip', os.path.join(drive_path, 'Dr.Face', convert_id+'.zip'))
      os.system('rm '+convert_id+'.zip')
      ##########print ('###############################' + 'save_workspace_data')
      time.sleep(3600)
def save_workspace_model():
  while 1:
    time.sleep(3600*2)
    #print ('jjkdhsjksjkdkdkdkdkldkdkdkdlld###############' + convert_id)
    
    
    f = open('/tmp/model.txt','r')
    convert_id = f.read()
    f.close()
    os.system('zip -ur workspace_'+convert_id+'.zip workspace/model'); os.system('cp workspace_'+convert_id+'.zip '+drive_path)
    
    
def get_preview(thr):
    #time.sleep(6000)
    if not os.path.isdir(datadir()): os.mkdir(datadir())
    if not os.path.isdir(datadir()+'/model'): os.mkdir(datadir()+'/model')
   #print (thread_list[0].pid)
   #print (psutil.pid_exists(thread_list[0].pid))
    
    while 1:
        
    
        if len(os.listdir(datadir()+'/model'))>5:
        
            #print ("getting ready to preview")
            
            os.system('rm -r '+datadir()+'/preview/merged')
            os.mkdir(datadir()+'/preview/merged')
        
            os.system("printf '0\nCPU\n' |  "+PYTHON_PATH+" DeepFaceLab/main.py merge --input-dir "+datadir()+"/preview --output-dir "+datadir()+"/preview/merged --output-mask-dir "+datadir()+"/preview/merged_mask --aligned-dir "+datadir()+"/preview/aligned --model-dir "+datadir()+"/model --model SAEHD --cpu-only  > /dev/null 2>&1")#> /dev/null 2>&1")
            os.system("printf '10\n' | "+PYTHON_PATH+" DeepFaceLab/main.py videoed video-from-sequence_  --input-dir "+datadir()+"/preview/merged --output-file "+datadir()+"/result_preview.mp4 > /dev/null 2>&1")#> /dev/null 2>&1")
            
            import moviepy.editor as mp
            if not os.path.isdir('assets'): os.mkdir('assets')
            
            if os.path.isfile(datadir()+"/result_preview.mp4"):
                
                text_trap = io.StringIO()
                sys.stdout = text_trap

                
                clip_resized = mp.VideoFileClip(datadir()+"/result_preview.mp4")
                #clip_resized = clip.resize(height=360) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
                convert_id_ = (''.join(map(choice,["bcdfghjklmnpqrstvwxz","aeiouy"]*3)))
                for filename in glob.glob("assets/*.mp4"):os.remove(filename)
                
                clip_resized.write_videofile("assets/result_preview"+convert_id_+".mp4")
                
               
                
                
            else:
            
                ##print ('No file found')
                time.sleep(60)
            
            
        else:
        
            time.sleep(60)
            
def put_msg(msg):
    #print (os.listdir('/tmp'))
    f = open('/tmp/processing','w+')
    f.write(msg)
    f.close()
def Main(q, option_id):
    
    ###########print ('############')
    ###########print (mode)
    import os
    files =os.listdir('/data')
    files.sort(key=lambda x: os.path.getmtime('/data/'+x))
    option_ = []#[{"label": '(1) New Wokspace', "value" : 1}, {"label": '(2) Resume Workspace', "value" : 1}, {"label": '(3) Load Workspace', "value" : 2, 'disabled': True}]
    for j,idx in enumerate(files[::-1]):
        option_.append({"label": idx , "value" : j+2})
  
    option_ = [{"label": '(1) New Workspace', "value" : 1}]+option_
    import os
    #global convert_id
    import time
    ###########print (option_)
    
    ##print (option_id)    
    ##print (option_)
    model = [i['label'] for i in option_ if i['value'] == int(option_id)][0]
    
    
    
    #print (option_id)
    
    if model == '(1) New Workspace':
        
        q.put('Initializing')
        put_msg('Initializing')
        #convert_id = (''.join(map(choice,["bcdfghjklmnpqrstvwxz","aeiouy"]*3)))
        
        
        for filename in glob.glob("/tmp/*npy"):os.remove(filename)
            
   
        #os.system('ln -s workspace '+os.path.join('/data', convert_id))
        
        if len(src_vids_clip)>0 and len(tar_vids_clip)>0:    
                
                
            
            #q.put('#ID-' + convert_id)
            
            #model_name = datadir()+'_'+convert_id + '.zip'
            
            q.put('[1/12] Loading Workspace')
            put_msg('[1/12] Loading Workspace')
            
            time.sleep(3)
      
            q.put  ('[2/12] Merging Source Videos')
            put_msg('[2/12] Merging Source Videos')
        
            
            source_files_merge = concatenate_videoclips(src_vids_clip)
            source_files_merge.write_videofile(datadir()+'/data_src.mp4',) 
                
            
                
            q.put  ('[3/12] Merging Target Videos')
            put_msg('[3/12] Merging Target Videos')
            
            target_files_merge = concatenate_videoclips(tar_vids_clip)
            target_files_merge.write_videofile(datadir()+'/data_dst.mp4',) 
        
                
          
            q.put  ('[4/12] Collecting frames from Source Videos')
            put_msg('[4/12] Collecting frames from Source Videos')
                
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py videoed extract-video --input-file "+datadir()+"/data_src.* --output-dir "+datadir()+"/data_src/ ", shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
            
            
                
                
            q.put  ('[5/12] Collecting frames from Target Videos')
            put_msg('[5/12] Collecting frames from Target Videos')
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py videoed extract-video --input-file "+datadir()+"/data_dst.* --output-dir "+datadir()+"/data_dst/", shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
            
            if p!= 0: q.put('Error while extracting Target frames! '); return False
           
            
            
                
                
                
            q.put  ('[6/12] Creating Source-face-profile')
            put_msg('[6/12] Creating Source-face-profile')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py extract --input-dir "+datadir()+"/data_src --output-dir "+datadir()+"/data_src/aligned --detector s3fd", shell=True).wait()
            
            if p!= 0: q.put('Error while extracting Source faces! '); return False
            
            
            q.put  ('[7/12] Creating Target-face-profile')
            put_msg('[7/12] Creating Target-face-profile')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py extract --input-dir "+datadir()+"/data_dst --output-dir "+datadir()+"/data_dst/aligned --detector s3fd", shell=True).wait()
            
            if p!= 0: q.put('Error while extracting Target faces! '); return False
            
            q.put  ('[8/12] Analyzing collected faces')
            put_msg('[8/12] Analyzing collected faces')
            
            print ('Analyzing source faces')
            
            os.mkdir('/tmp/cluster')
            labelsdict = {}
            labelsdict['src_face_labels'] = {}
            labelsdict['dst_face_labels'] = {}
            
            labelsdict['src_face_labels'] = ffc.Get_face_clustered_labels(datadir()+'/data_src/aligned')
            print ('Analyzing target faces')
            labelsdict['dst_face_labels'] = ffc.Get_face_clustered_labels(datadir()+'/data_dst/aligned')
            
            
            
            np.save('/tmp/cluster/labelsdict.npy', labelsdict) 
            
            if len(labelsdict['src_face_labels']) >1 or len(labelsdict['dst_face_labels']) >1:
                q.put  ('[8/12] Mutiple person detected. Manually add faces to continue.')
                put_msg('[8/12] Mutiple person detected. Manually add faces to continue.')
            
            from numba import cuda 
            device = cuda.get_current_device()
            device.reset()
            
            #run.value = 1
            
            
            while True:
            
                if os.path.isdir('/tmp/cluster'):
                
                    time.sleep(4)
                    
                else:
                
                    break
            os.system(PYTHON_PATH+' DeepFaceLab/preview.py')
            
            
            
            
            q.put  ('[9/12] Enhancing Source faces')
            put_msg('[9/12] Enhancing Source faces')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_src/aligned", shell=True).wait()
            
            if p!= 0: q.put('Error while enhancing Source faces! '); return False
            
            
            q.put  ('[10/12] Enhancing Target faces')
            put_msg('[10/12] Enhancing Target faces')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_dst/aligned", shell=True).wait()
            
            if p!= 0: q.put('Error while enhancing Target faces! '); return False
            
            
            
            
            #q.put  ('Enhancing Faces ')
            
            #p = [subprocess.Popen("echo | python DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_src/aligned", shell=True),
            #    subprocess.Popen("echo | python DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_dst/aligned", shell=True)]
            #p_ = [p[0].wait(), p[1].wait()]
            
            #if p_[0] != 0 and p_[1]!= 0: 
            #    q.put('Error while Enhancing faces! ')
            #    return False    
            
                            
            print ('Creating Segmentation mask')
            q.put  ('[11/12] Generating face segmentation masks')
            put_msg('[11/12] Generating face segmentation masks')
            
            p = os.system(PYTHON_PATH+' face_seg.py')
            if p != 0: 
                q.put('Error while extracting face masks! ')
                return False
            q.put  ('[12/12] Preparing to start Training')
            put_msg('[12/12] Preparing to start Training')
            import os
            #os.chdir("/content")
            import  os, time
            
            
            
            q.put('Training In Progress')
            if os.path.isfile('/tmp/processing'):os.remove('/tmp/processing')
            
        
        
             
            p = os.system('echo | '+PYTHON_PATH+' DeepFaceLab/main.py train --training-data-src-dir '+datadir()+'/data_src/aligned --training-data-dst-dir '+datadir()+'/data_dst/aligned --pretraining-data-dir pretrain --model-dir '+datadir()+'/model --model SAEHD')
            
            #print ("")
            
            q.put(':Stopped:')
                
                
            return True
            
        else:
        
            q.put(':Stopped:')
            return False
        
        
    
    elif  int(option_id)>1:
    
        #print (option_id)
    
        q.put('Initializing')
        put_msg('Initializing')
        convert_id = model#.split(datadir()+'_')[-1].split('.')[0]
        
        #if not os.path.isfile('/tmp/model.txt'):
        #convert_id = (''.join(map(choice,["bcdfghjklmnpqrstvwxz","aeiouy"]*3)))
        
        #os.system('rm -r '+datadir()+'')
        
        #os.system('ln -s workspace '+os.path.join('/data', convert_id))
        #print ('################################3')
        
        for filename in glob.glob("assets/*.mp4"):os.remove(filename)
        #print ('################################3')         
        if not os.path.isdir('assets'): os.mkdir('assets')
        #print ('################################3')
        if os.path.isfile(datadir()+'/result_preview.mp4'):
            copyfile(datadir()+'/result_preview.mp4', 'assets/result_preview.mp4')
        #print ('################################3')
        
        q.put('#ID-' + convert_id)
        
        
        
        #model_name = datadir()+'_'+convert_id + '.zip'
        #if os.path.isfile(datadir()+'/data_dst.mp4') and os.path.isfile(datadir()+'/data_src.mp4'):
        
        if os.path.isfile(datadir()+'/model/iteration.txt'):
        
            q.put('[1/2] Downloading Model' )
            put_msg('[1/2] Downloading Model' )
    
            #print ('################################3')
                    
            os.system(PYTHON_PATH+' DeepFaceLab/preview.py')
                    
            
            q.put('[2/2] Loading Workspace')
            put_msg('[2/2] Loading Workspace')
            
            #print ('################################3')
            import os
            #os.chdir("/content")
            q.put('Training In Progress')
            if os.path.isfile('/tmp/processing'):os.remove('/tmp/processing')
            p = os.system('echo | '+PYTHON_PATH+' DeepFaceLab/main.py train --training-data-src-dir '+datadir()+'/data_src/aligned --training-data-dst-dir '+datadir()+'/data_dst/aligned --pretraining-data-dir pretrain --model-dir '+datadir()+'/model --model SAEHD')
            q.put(':Stopped:')
                
            return True
            
        elif os.path.isfile(datadir()+'/data_dst.mp4') and os.path.isfile(datadir()+'/data_src.mp4'):
            
            q.put('[1/10] Loading Workspace')
            put_msg('[1/10] Loading Workspace')
        
            q.put  ('[2/10] Collecting frames from Source Videos')
            put_msg('[2/10] Collecting frames from Source Videos')
                
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py videoed extract-video --input-file "+datadir()+"/data_src.* --output-dir "+datadir()+"/data_src/ ", shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
            
            
                
                
            q.put  ('[3/10] Collecting frames from Target Videos')
            put_msg('[3/10] Collecting frames from Target Videos')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py videoed extract-video --input-file "+datadir()+"/data_dst.* --output-dir "+datadir()+"/data_dst/", shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
            
            if p!= 0: q.put('Error while extracting Target frames! '); return False
           
            
            
                
                
                
            q.put  ('[4/10] Creating Source-face-profile')
            put_msg('[4/10] Creating Source-face-profile')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py extract --input-dir "+datadir()+"/data_src --output-dir "+datadir()+"/data_src/aligned --detector s3fd", shell=True).wait()
            
            if p!= 0: q.put('Error while extracting Source faces! '); return False
            
            
            q.put  ('[5/10] Creating Target-face-profile')
            put_msg('[5/10] Creating Target-face-profile')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py extract --input-dir "+datadir()+"/data_dst --output-dir "+datadir()+"/data_dst/aligned --detector s3fd", shell=True).wait()
            
            if p!= 0: q.put('Error while extracting Target faces! '); return False
            
            q.put  ('[6/10] Analyzing collected faces')
            put_msg('[6/10] Analyzing collected faces')
            
            print ('Analyzing source faces')
            os.mkdir('/tmp/cluster')
            labelsdict = {}
            labelsdict['src_face_labels'] = {}
            labelsdict['dst_face_labels'] = {}
            
            labelsdict['src_face_labels'] = ffc.Get_face_clustered_labels(datadir()+'/data_src/aligned')
            print ('Analyzing target faces')
            labelsdict['dst_face_labels'] = ffc.Get_face_clustered_labels(datadir()+'/data_dst/aligned')
            
            
                
            np.save('/tmp/cluster/labelsdict.npy', labelsdict) 
            
            if len(labelsdict['src_face_labels']) >1 or len(labelsdict['dst_face_labels']) >1:
                q.put  ('[8/12] Mutiple person detected. Manually add faces to continue.')
                put_msg('[8/12] Mutiple person detected. Manually add faces to continue.')
            
            from numba import cuda 
            device = cuda.get_current_device()
            device.reset()
            
            #run.value = 1
            
            
            while True:
            
                if os.path.isdir('/tmp/cluster'):
                
                    time.sleep(4)
                    
                else:
                
                    break
            os.system(PYTHON_PATH+' DeepFaceLab/preview.py')
            
            
            
            
            q.put  ('[7/10] Enhancing Source faces')
            put_msg('[7/10] Enhancing Source faces')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_src/aligned", shell=True).wait()
            
            if p!= 0: q.put('Error while enhancing Source faces! '); return False
            
            
            q.put  ('[8/10] Enhancing Target faces')
            put_msg('[8/10] Enhancing Target faces')
            
            p = subprocess.Popen("echo | "+PYTHON_PATH+" DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_dst/aligned", shell=True).wait()
            
            if p!= 0: q.put('Error while enhancing Target faces! '); return False
            
            
            
            
            #q.put  ('Enhancing Faces ')
            
            #p = [subprocess.Popen("echo | python DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_src/aligned", shell=True),
            #    subprocess.Popen("echo | python DeepFaceLab/main.py facesettool enhance --input-dir "+datadir()+"/data_dst/aligned", shell=True)]
            #p_ = [p[0].wait(), p[1].wait()]
            
            #if p_[0] != 0 and p_[1]!= 0: 
            #    q.put('Error while Enhancing faces! ')
            #    return False    
            
                            
            print ('Creating Segmentation mask')
            q.put  ('[9/10] Generating face segmentation masks')
            put_msg('[9/10] Generating face segmentation masks')
            
            p = os.system(PYTHON_PATH+' face_seg.py')
            if p != 0: 
                q.put('Error while extracting face masks! ')
                return False
            q.put  ('[10/10] Preparing to start Training')
            put_msg('[10/10] Preparing to start Training')
            import os
            #os.chdir("/content")
            import  os, time
            
            
            
            q.put('Training In Progress')
            if os.path.isfile('/tmp/processing'):os.remove('/tmp/processing')
             
            p = os.system('echo | '+PYTHON_PATH+' DeepFaceLab/main.py train --training-data-src-dir '+datadir()+'/data_src/aligned --training-data-dst-dir '+datadir()+'/data_dst/aligned --pretraining-data-dir pretrain --model-dir '+datadir()+'/model --model SAEHD')
            
            #print ("")
            
            q.put(':Stopped:')
                
                
            return True
            
        else:
        
            q.put(':Stopped:')
            return False        
            
            
        
          
            
import os
import logging
server = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = dash.Dash(__name__, server=server, update_title=None, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.7.2/css/all.css"])
app.title = 'dr.face'
app.update_title = None
server = app.server
global slider_prev_instance 
slider_prev_instance = [0,1000]
global threadon 
threadon = True
global threadon_ 
threadon_ = True
global gui_queue
gui_queue = Queue() 
global slider_prev_instance2 
slider_prev_instance2 = [0,1000]
global storemsg
storemsg= ''
global start
start = ''
global tt
tt = False
global watch
watch = stopWatch()
global tt1
tt1 = False
global tt2
tt2 = False
global msglist 
global HEIGHT
HEIGHT = 256
global src_vids
src_vids = []
global tar_vids
tar_vids = []
msglist = 'Initializing'
global src_vids_clip
src_vids_clip = []
global tar_vids_clip
tar_vids_clip = []
global horizontal_shear
global vertical_shear
global horizontal_shift
global vertical_shift
global ind_preview
horizontal_shear = 0
vertical_shear = 0
horizontal_shift = 0
vertical_shift = 0
ind_preview = 0
if not os.path.isdir('videos'): os.mkdir('videos')
if not os.path.isdir('videos/Source'): os.mkdir('videos/Source')
if not os.path.isdir('videos/Source/Youtube'): os.mkdir('videos/Source/Youtube')
if not os.path.isdir('videos/Source/Upload'): os.mkdir('videos/Source/Upload')
if not os.path.isdir('videos/Source/Record'): os.mkdir('videos/Source/Record')
if not os.path.isdir('videos/Source/Final'): os.mkdir('videos/Source/Final')
if not os.path.isdir('videos/Target'): os.mkdir('videos/Target')
if not os.path.isdir('videos/Target/Youtube'): os.mkdir('videos/Target/Youtube')
if not os.path.isdir('videos/Target/Upload'): os.mkdir('videos/Target/Upload')
if not os.path.isdir('videos/Target/Record'): os.mkdir('videos/Target/Record')
if not os.path.isdir('videos/Target/Final'): os.mkdir('videos/Target/Final')
  
record = [html.Div(children = [html.Img(src="/video_feed", style={
            'width': '266px',
            'height': '200px'
            }), html.Hr(), dbc.Button("Start", outline=True, color="primary", className="mr-1", id='rec_button')])]
        
def loading(children):
  return dcc.Loading(children, type='dot', fullscreen=False, style={'opacity': 0.2})    
def video_index():
  global src_vids_clip
  
  return len(src_vids_clip)
def video_index2():
  global tar_vids_clip
  return len(tar_vids_clip)
def duration():
  global src_vids_clip
  return int(sum([i.duration for i in src_vids_clip]))
def duration2():
  global tar_vids_clip
  return int(sum([i.duration for i in tar_vids_clip]))
  
def get_timeago(dirname):
    #print(dirname)
    f = time.time() - os.path.getmtime(dirname)#((max(time.time()-os.stat(root).st_mtime for root,_,_ in os.walk(dirname))))
    now = datetime.datetime.now() + datetime.timedelta(seconds = f)
    date = datetime.datetime.now()
    #print (date)
    return  (timeago.format(date, now)) # will #print 3 minutes ago
      
import glob
import os
search_dir = "/data/"
# remove anything from the list that is not a file (directories, symlinks)
# thanks to J.F. Sebastion for pointing out that the requirement was a list 
# of files (presumably not including directories)  
files =os.listdir('/data')
files.sort(key=lambda x: os.path.getmtime('/data/'+x))
#print (files)   
global option_  
option_ = []#[{"label": '(1) New Wokspace', "value" : 1}, {"label": '(2) Resume Workspace', "value" : 1}, {"label": '(3) Load Workspace', "value" : 2, 'disabled': True}]
for j,idx in enumerate(files[::-1]):
    option_.append({"label": idx , "value" : j+2})
    
option__ = [{"label":'/data/'+i["label"]  + ' - modified '+get_timeago('/data/'+i["label"] ), "value":i["value"]} for i in option_]


files_ = os.listdir('/data')

global option_import  
option_import = []
for j,idx in enumerate(files_[::-1]):
        if os.path.isdir(os.path.join('/data',idx,'model')):
            if len(os.listdir(os.path.join('/data',idx,'model')))>4:
                option_import.append({"label": '/data/'+idx , "value" : j+2})

if IN_COLAB_DRIVE:
    zipfiles = os.listdir(os.path.join(drive_path,'Dr.Face'))
    option_drive = []
    for j,idx in enumerate(zipfiles):
        option_drive.append({"label": idx , "value" : j+1})
        
    option_drive_ = [{"label":'[DRIVE] '+os.path.join(drive_path, 'Dr.Face',i["label"]), "value":i["value"]} for i in option_drive]
n_options = len(option_)
if n_options == 0:
    ow_disabled = True
    
else:
    ow_disabled = False
    

if len(option_import)>0:
    ll = [{'label' : 'Training from scratch', 'value' : '0'}, {'label' : 'Import a model', 'value' : '1'}]
    
else:
    ll =  [{'label' : 'Training from scratch', 'value' : '0'}, {'label' : 'Import a model', 'value' : '1', 'disabled':True}]
        
        
GPUs_opts = [{"label":'CPU', "value":'C'}]+[{"label": i.name+' ['+str(int(i.memoryTotal))+' MB]',"value": i.id} for i in  GPUtil.getGPUs()]
New_modal = html.Div([html.Br(),
dbc.Input(placeholder="Enter Workspace Name", id="start_text_new"),
html.Br(),

dbc.FormGroup([
 
        dbc.Label("Select Training Type:"),
        dbc.RadioItems(
            options=ll,
            value='0',
            id="select_train_scratch",
            inline=True,
        ),]),
        
        html.Div([dbc.Select(id = 'option_import_select', options = option_import, value = 1), html.Br(),  html.Br()], id = 'option_import_select_div',style = {'display':'none'}),
        

dbc.FormGroup([
 
        dbc.Label("Select Mode:"),
        dbc.RadioItems(
            options=[{'label' : 'Face & Head', 'value' : '0'}, {'label' : 'Full face', 'value' : '1'}, {'label' : 'Half Face', 'value' : '2'}],
            value='1',
            id="select_mode",
            inline=True,
        ),], id = "select_mode_form"),
        
 dbc.FormGroup([
 
        dbc.Label("Select Quality:"),
        dbc.RadioItems(
            options=[
                {"label": "High", "value": 1},
                {"label": "Medium", "value": 2},
                {"label": "Low", "value": 3},
                {"label": "Very Low", "value": 4},
            ],
            value=2,
            id="select_resolution",
            inline=True,
        ),],
    
id = "select_resolution_form"),
 dbc.FormGroup(
    [
        dbc.Label("Select Device:"),
        dbc.Checklist(
            options=GPUs_opts,
            value=GPUs_opts[0]['value'],
            id="select_device",
            inline=True,
        ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Select Batchsize:"),
        dbc.RadioItems(
            options=[
                {"label": "Auto", "value": 1},
                {"label": "2", "value": 2},
                {"label": "4", "value": 4},
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
            ],
            value=1,
            id="select_Batchsize",
            inline=True,
        ),
    ]
)
])
if IN_COLAB_DRIVE: start_text_input_disp = {'display':''}
else: start_text_input_disp = {'display':'none'}
Open_modal = html.Div([html.Br(),
dbc.InputGroup([dbc.Select(id = 'start_text_input_', options = option__, value = 2), dbc.Button(outline=True, id = 'drive_dload', active=False, disabled = False, color="primary", className="fab fa-google-drive",style = start_text_input_disp)]),
html.Br(),
 dbc.FormGroup(
   
),
 dbc.FormGroup(
    [
        dbc.Label("Select Device:"),
        dbc.Checklist(
            options=GPUs_opts,
            value=GPUs_opts[0]['value'],
            id="select_device_",
            inline=True,
        ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Select Batchsize:"),
        dbc.RadioItems(
            options=[
                {"label": "Auto", "value": 1},
                {"label": "2", "value": 2},
                {"label": "4", "value": 4},
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
            ],
            value=1,
            id="select_Batchsize_",
            inline=True,
        ),
    ]
)
])
Progress_modal = html.Div([html.Div(id = 'progress_msg'), html.Br(),dbc.Progress(value=0, id="Progress_modal", striped=True, animated = True)])
option_ = [{"label": '(1) New Workspace', "value" : 1}]+option_
Progress =  html.Div([html.Div([dbc.Button(' New',outline=False, id = 'New_workspace',  active=False, disabled = True, color="light", className="fas fa-plus",),
dbc.Button(' Open',outline=False, id = 'Open_workspace', active=False, disabled = False,color="light", className="fas fa-redo")], id = 'start_buttons', style = {'text-align' : 'center', 'color':'blue'})
,dbc.Modal(
            [
                dbc.ModalHeader("Create a New Workspace"),
                dbc.ModalBody(New_modal),
                dbc.ModalFooter(
                    dbc.Button(
                        "Confirm", id="New_modal_Butt", className="ml-auto"
                    )
                ),
            ],
            id="New_modal_",
            centered=True,
          
        ),
        
        
        dbc.Modal(
            [
                dbc.ModalHeader("Open an Workspace"),
                dbc.ModalBody(Open_modal),
                dbc.ModalFooter(
                    dbc.Button(
                        "Confirm", id="Open_modal_Butt", className="ml-auto", disabled = False
                    )
                ),
            ],
            id="Open_modal_",
            centered=True,
          
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Processing Datasets"),
                dbc.ModalBody([Progress_modal,html.Br(), html.Div(id = 'choose_face_modal', style = {'text-align':'center'})
                , ]),
                
          
                
            ],
            id="Progress_modal_",
            centered=True,
            backdrop = 'static',
            autoFocus = True
          
        ),
        
        dbc.Modal(    
            [   dcc.Loading(html.Div(id="drive_dload_loading"), type='dot', fullscreen=True, style={'opacity': 0.8}),
                dbc.ModalHeader("Download Workspace from Drive"),
                dbc.ModalBody([dbc.Select(id = 'drive_dload_input', options = option_drive_, value = 1)]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Download", id="drive_dload_butt", className="ml-auto", disabled = False
                    )
                ),
          
                
            ],
            id="drive_dload_",
            centered=True,
         #   backdrop = 'static',
            autoFocus = True
          
        ),
        
        
        dbc.Modal(    
            [   
                dbc.ModalHeader('In Progress', id = 'ModalHeader_convert'),
                dbc.ModalBody([html.Div(id = 'convert_result'),
                                html.Br(),
                                dbc.Progress(id = 'merge_progress')]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Finish", id="merge_progress_exit", className="ml-auto", disabled = False, style = {'display':'none'}
                    )
                ),
          
                
            ],
            id="merge_progress_modal",
            centered=True,
            backdrop = 'static',
            autoFocus = True
          
        ),
        
        
        
        
        
        
        
        dbc.Modal(
            [
                dbc.ModalHeader("ResourceExhaustedError! Here are some workarounds."),
                dbc.ModalBody('You don\'t have enough GPU memeory. Please run the program in "low quality" mode and select the "Auto" batchsize.'),
                
          
                
            ],
            id="error_modal",
            #centered=True,
            backdrop = 'static',
            #autoFocus = True
          
        ),
        
        
        html.Div(dbc.InputGroup(
            [dbc.InputGroupAddon("Model", addon_type="prepend"), dbc.Input(id = 'start_text_input'), dbc.Select(id = 'face_type_select', 
            options = [{'label' : 'Head', 'value' : '0'}, {'label' : 'Full face', 'value' : '1'}, {'label' : 'Face', 'value' : '2'}], value = '1'),
            dbc.Button(outline=True, id = 'start_text_continue', active=False, disabled = False, color="success", className="fas fa-check-circle")
    ], 
            size="sm",
        ), style = {'display':'none'}),
        
        html.Div([html.Br(), html.Div(id = 'preview_graph'), html.Br(),html.Div(id = 'preview_imgs'),dbc.Progress(id = 'preview_progress', style={"height": "6px"})], id= 'preview_divs', style = {'display':'none'})])
        
       
#dbc.InputGroup(
#            [dbc.InputGroupAddon("Model", addon_type="prepend"), dbc.Select(id = 'start_text_input', options = option_, value = '0'), dbc.Select(id = 'face_type_select', 
#            options = [{'label' : 'Head', 'value' : '0'}, {'label' : 'Full face', 'value' : '1'}, {'label' : 'Face', 'value' : '2'}], value = '1'),
#            dbc.Button(outline=True, id = 'start_text_continue', active=False, disabled = False, color="success", className="fas fa-check-circle")
#    ], 
#            size="sm",
#        ), html.Br(), html.Div(id = 'preview_graph'), html.Br(),loading(html.Div(id = 'preview_imgs'))]), #dcc.RadioItems(id = 'Progress_select', value = ''), html.Hr(id = 'hr2'), 
#dbc.Button('Continue', size="sm", id = 'start_text_continue'),  
#html.Hr(id = 'hr3'), html.Div(id = 'progress_field')]
try:
    url=os.path.join('assets', glob.glob("assets/*mp4")[0].split('/')[-1])
except:
    url = ''
Preview_vid = html.Div([dash_player.DashPlayer(
    id='video-player',
    #url='http://media.w3.org/2010/05/bunny/movie.mp4',
    url = url,
    controls=True, loop = True, playing = True, width='100%', height='100%', 
),
])
    
    
size_layout  = dbc.Card(
    dbc.CardBody(
    
    
        [   dbc.Row(daq.Slider(min=0,max=50,value=10,step=1, id = "size_step", size = 150)),
        
            html.Hr(),
          
            dbc.Row(dbc.Button(outline=True, id = 'v_plus_size', active=False, disabled = convert_disabled, color="success", className="fas fa-plus-circle"),justify="center",),
          
            dbc.Row([dbc.Col(dbc.Button(outline=True, id = 'h_minus_size', active=False, disabled = convert_disabled, color="success", className="fas fa-minus-circle")),  dbc.Col(dbc.Button(outline=True, id = 'h_plus_size', active=False, disabled = convert_disabled, color="success", className="fas fa-plus-circle"))]),
      
            dbc.Row(dbc.Button(outline=True, id = 'v_minus_size', active=False, disabled = convert_disabled, color="success", className="fas fa-minus-circle"), justify="center"),
            
            dbc.Tooltip('Stretch face vertically', target="v_plus_size"),
            dbc.Tooltip('Shrink face horizontally', target="h_minus_size"),
            dbc.Tooltip('Stretch face horizontally', target="h_plus_size"),
            dbc.Tooltip('Shrink face vertically', target="v_minus_size"),
        ]
    ),
    style={"width": "10rem"},
)
shift_layout = dbc.Card(
    dbc.CardBody(
        [   dbc.Row(daq.Slider(min=0,max=50,value=10,step=1, id = "shift_step", size = 150)),
            
            html.Hr(),
            
            dbc.Row(dbc.Button(outline=True, id = 'v_plus_shift', active=False, disabled = convert_disabled, color="success", className="fas fa-chevron-circle-up"),justify="center",),
          
            dbc.Row([dbc.Col(dbc.Button(outline=True, id = 'h_minus_shift', active=False, disabled = convert_disabled, color="success", className="fas fa-chevron-circle-left")),  dbc.Col(dbc.Button(outline=True, id = 'h_plus_shift', active=False, disabled = convert_disabled, color="success", className="fas fa-chevron-circle-right"))]),
      
            dbc.Row(dbc.Button(outline=True, id = 'v_minus_shift', active=False, disabled = convert_disabled, color="success", className="fas fa-chevron-circle-down"), justify="center"),
            
            dbc.Tooltip('Move face upward', target="v_plus_shift"),
            dbc.Tooltip('Move face to the left', target="h_minus_shift"),
            dbc.Tooltip('Move face to the right', target="h_plus_shift"),
            dbc.Tooltip('Move face downward', target="v_minus_shift"),
        ]
    ),
    style={"width": "10rem"},
)
basic_Set = [ html.Br(),
dbc.Row([(dbc.InputGroup([dbc.InputGroupAddon("Mask type", addon_type="prepend"),dbc.Select(id = 'mask_mode_', options = [{'label':'dst', "value" :1},
{'label':'learned-prd', "value" :2}, 
{'label':'learned-dst', "value" :3}, 
{'label':'learned-prd*learned-dst', "value" :4}, 
{'label':'learned-prd+learned-dst', "value" :5},  
], value = 3),], size="sm",)), 
(dbc.InputGroup([dbc.InputGroupAddon("Mode", addon_type="prepend"),dbc.Select(id = 'mode_', options = [{'label':'original', "value" :'original'},
{'label':'overlay', "value" :'overlay'}, 
{'label':'hist-match', "value" :'hist-match'}, 
{'label':'seamless', "value" :'seamless'}, 
{'label':'seamless-hist-match', "value" :'seamless-hist-match'},  
{'label':'raw-rgb', "value" :'raw-rgb'}, 
{'label':'raw-predict', "value" :'raw-predict'}], value = 'overlay') ], size="sm",)),
(dbc.InputGroup([dbc.InputGroupAddon("Color mode", addon_type="prepend"),dbc.Select(id = 'color_mode_', options = [{'label':'None', "value" :0},
{'label':'rct', "value" :1}, 
{'label':'lct', "value" :2}, 
{'label':'mkl', "value" :3}, 
{'label':'mkl-m', "value" :4},  
{'label':'idt', "value" :5}, 
{'label':'idt-m', "value" :6},  
{'label':'sot-m', "value" :7}, 
{'label':'mix-m', "value" :8}], value = '0')], size="sm"))], justify = 'center',  no_gutters=True,),
    
            ]
            
            
adv_set = [dbc.CardBody([
  dcc.Slider(
  min=0,
  max=100,
  value=0,
  step=1,
  id = "motion_blur_power_", marks = {0: '0', 100:'100', 50: 'Motion Blur Power'}
),
html.Br(),
dcc.Slider(
  min=-400,
  max=400,
  value=0,
  step=1,
  id = "Erode_" , marks = {-400: '-400', 0:'Erode', 400:'+400'}
)
,
html.Br(),
  dcc.Slider(
  min=0,
  max=400,
  value=0,
  step=1,
  id = "Blur_", marks = {0: '0', 200:'Blur', 400:'+400'}
)
,
html.Br(),
  dcc.Slider(
  min=-100,
  max=100,
  value=0,
  step=1,
  id = "blursharpen_amount_",  marks = {-100: '-100', 100:'100', 0: 'Blur-sharpen Amount'}
)
,
html.Br(),
  dcc.Slider(
  min=0,
  max=500,
  value=0,
  step=1,
  id = "image_denoise_power_",  marks = {0: '0', 500:'500', 250: 'Image Denoise Power'}
)
,   
html.Br(),
  dcc.Slider(
  min=0,
  max=100,
  value=0,
  step=1,
  id = "color_degrade_power_",  marks = {0: '0', 100:'100', 50: 'Color Degrade Power'}
)])]
Convert_Tab =  html.Div([dbc.Card(
    [   
        
        #.CardHeader(dbc.InputGroup(
        #        [dbc.InputGroupAddon("Model", addon_type="prepend"), dbc.Select(id = 'convert_model_id', options = option_convert, value = '0'), 
        #        dbc.Button(outline=True, id = 'convert_model_continue', active=False, disabled = False, color="success", className="fas fa-check-circle")
        #, dbc.Button(outline=True, id = 'refresh_img', active=False, disabled = convert_disabled, color="primary", className="fas fa-redo"), 
        #dbc.Button(outline=True, id = 'okay_merge', active=False, disabled = convert_disabled, color="danger", className="fas fa-sign-in-alt")], 
        #        size="sm",
        #    )),New_workspace
    # 
    #    html.Div(id = 'convert_result', style = {'text-align' : 'center'}),
      #   html.Div(id = 'convert_load', style = {'text-align' : 'center'}),
      #   html.Hr(),
      
      
      loading([html.Div(dbc.ButtonGroup(
            [
            
            dbc.Button( id = 'default_pre', active=False, disabled = convert_disabled, color="light", size = 'sm', className="far fa-window-maximize"),
                dbc.Button( id = 'ori_pre', active=False, disabled = convert_disabled, color="light", size = 'sm', className="fas fa-window-maximize"),
                dbc.Button( id = 'default_face_pre', active=False, disabled = convert_disabled, color="light", size = 'sm', className="fas fa-smile-beam"),
                dbc.Button( id = 'ori_face_pre', active=False, disabled = convert_disabled, color="light", size = 'sm', className="far fa-smile-beam"),
                
            dbc.Button( id = 'refresh_img', active=False, disabled = convert_disabled, color="light", size = 'sm', className="fas fa-redo"),
            
                dbc.Button( id = 'okay_merge', active=False, disabled = convert_disabled, color="light", size = 'sm', className="fas fa-sign-in-alt"),
                
                ]),
                style = {'text-align':'center', 'margin-bottom':'-23px'}) ,
        
        dbc.CardImg(top=True, id = 'Convert_Image'),
        
        #dbc.Progress(id = 'merge_progress'),
        
        #dcc.Loading(html.Div(id = 'test_div'), type = 'dot'),
        
        #dcc.Loading(html.Div('  ', id = 'test_div'), type = 'circle'),
        
        
        
    #dbc.Tooltip('Choose Workspace', target="convert_model_continue"),
    #dbc.Tooltip('Refresh Preview Image', target="refresh_img"),
    #dbc.Tooltip('Convert', target="okay_merge"),
        
        html.Div(dbc.ButtonGroup(
            [dbc.Button( id = 'v_plus_size', active=False, disabled = convert_disabled, color="light",size = 'sm', className="fas fa-plus-circle"),
            dbc.Button( id = 'h_minus_size', active=False, disabled = convert_disabled, color="light",size = 'sm', className="fas fa-minus-circle"),
            dbc.Button( id = 'h_plus_size', active=False, disabled = convert_disabled, color="light",size = 'sm', className="fas fa-plus-circle"),
            dbc.Button( id = 'v_minus_size', active=False, disabled = convert_disabled, color="light",size = 'sm', className="fas fa-minus-circle"),
            dbc.Button( id = 'v_plus_shift', active=False, disabled = convert_disabled, color="light", size = 'sm',className="fas fa-chevron-circle-up"),
            dbc.Button( id = 'h_minus_shift', active=False, disabled = convert_disabled, color="light",size = 'sm', className="fas fa-chevron-circle-left"),    
            dbc.Button( id = 'h_plus_shift', active=False, disabled = convert_disabled, color="light",size = 'sm', className="fas fa-chevron-circle-right"),
            dbc.Button( id = 'v_minus_shift', active=False, disabled = convert_disabled, color="light",size = 'sm',className="fas fa-chevron-circle-down"),
            ]), style = {'text-align':'center', 'margin-top':'-23px'})]) ,
    dbc.Tabs(
            [
                dbc.Tab(basic_Set, label="Basic", tab_id="Basic-1"),
                dbc.Tab(adv_set, label="Advanced", tab_id="Advanced-1"),
                
        
            ],
            id="settings_tabs",
            active_tab="Basic-1",
        ),
        html.Br(),
        
        html.Div(dbc.Button('Swap Full Video', id = 'convert_start', size = 'sm', active=True, color="light"), style = {'text-align' : 'center'}),
#html.Br(),    
#dbc.Row([dbc.Col(src_child_), dbc.Col(dst_child_)], justify = 'center')
  
    
      dbc.Tooltip('Stretch face vertically', target="v_plus_size"),
            dbc.Tooltip('Shrink face horizontally', target="h_minus_size"),
            dbc.Tooltip('Stretch face horizontally', target="h_plus_size"),
            dbc.Tooltip('Shrink face vertically', target="v_minus_size"),       
            
            dbc.Tooltip('Move face upward', target="v_plus_shift"),
            dbc.Tooltip('Move face to the left', target="h_minus_shift"),
            dbc.Tooltip('Move face to the right', target="h_plus_shift"),
            dbc.Tooltip('Move face downward', target="v_minus_shift"),
            dbc.Tooltip('Refresh Image', target="refresh_img"),
            dbc.Tooltip('Save Settings', target="okay_merge"),
            
            dbc.Tooltip('Resultant frame', target="default_pre"),
            dbc.Tooltip('Original frame', target="ori_pre"),
            dbc.Tooltip('Resultant face', target="default_face_pre"),
            dbc.Tooltip('Original face', target="ori_face_pre"),
            
        ],
        
    ),
])
final_convert = dbc.Jumbotron(
[
dbc.Container(
    [
        #html.P(
        #    "The conversion process will take some time. ", className="lead",
        
        html.Div(html.P('The conversion process will take some time', className="lead") , style = {'text-align' : 'center'}),
        html.Br(),
        html.Div(dbc.Button('Convert', id = 'convert_start', active=True, color="primary"), style = {'text-align' : 'center'}),
        
        html.Div(id = 'convert_result', style = {'text-align' : 'center'}),
        html.Br(),
        dbc.Progress(id = 'merge_progress')
    ],
    fluid=True,
)
],
fluid=True,
)
right_frame = dbc.Tabs(
            [
                dbc.Tab(Preview_vid, label="Preview", tab_id="Preview_vid"),
                dbc.Tab(Convert_Tab, label="Settings", tab_id="Convert_Tab"),
                #dbc.Tab(final_convert, label="Convert", tab_id="final_Tab"),
        
            ],
            id="convert_tabs_1",
            active_tab="Preview_vid",
        )
#dbc.Row([dbc.Col(html.Button('Button 1', id='btn-nclicks-1')), dbc.Col(html.Button('Button 2', id='btn-nclicks-2'))], justify = 'center')
#dbc.ButtonGroup(
#            [dbc.Button(outline=True, id = 'convert_settings', active=False, disabled = False, color="success", className="fas fa-hourglass-start"),
#dbc.Button('Convert', id = 'convert_', active=False, disabled = False, color="danger")],
#           
#            className="mr-1",  style = {'text-align' : 'center'}),
            
#Images = loading([
#dbc.Tabs(
  #   [
#        dbc.Tab(html.Img(id = 'Face', style = {'width' : '100%', 'height' : '100%'}), label="Images-1"),
#        dbc.Tab(html.Img(id = 'Mask', style = {'width' : '100%', 'height' : '100%'}), label="Images-2"),
#
#    ]), dbc.Button(outline=True, id = 'Images-refresh', active=False, disabled = False, color="success", className="fas fa-redo-alt")]
#)#[(html.Div(id = 'ImagesG'))]
#Result = [(html.Div(id = 'Result_out'))]
try:
    sec_s = ' [Updated ' +str(int(time.time() - os.path.getctime(glob.glob("assets/*mp4")[0]))//60)  + ' minutes ago]'
    
    
except:
    sec_s = ''
    
choose_face = html.Div([html.Div(id = 'all_imgs_faces'), html.Br(),
html.Div(id = 'okay_face_select_text')])
controls_start = dbc.Jumbotron(
    [
        html.H1("Start the Process", id  = 'status'),
    dbc.Tooltip( id = 'status_tooltip',
            target="status",
            placement = 'left'
        ),
        html.Br(),
        dbc.ButtonGroup(
            [dbc.Button(outline=True, id = 'Start-click', active=False, disabled = False, color="success", className="fas fa-hourglass-start"),
#dbc.Button(outline=True, id = 'Images-addclick', active=False,disabled = True, color="primary", className="fas fa-image"),
#dbc.Button(outline=True, id = 'Settings-addclick', active=False,disabled = False, color="primary", className="fas fa-users-cog"),
dbc.Button(outline=True, id = 'Resetal-addclick', active=False, disabled = False, color="danger", className="fas fa-power-off"),
dbc.Button(outline=True, id = 'delete-addclick', active=False, disabled = False, color="danger", className="fas fa-trash-alt")],
            
            className="mr-1"),
            
        html.Hr(className="my-2"),
        
        
        
        dbc.Row([dbc.Col(dbc.Toast(Progress, id="toggle-add-Progress",header="Getting Started",is_open=True,icon="primary",dismissable=True,  style={"maxWidth": "1000px"})),
        dbc.Col(dbc.Toast(right_frame, id="toggle-add-right_frame",header=""+sec_s,is_open=False,icon="primary",dismissable=True,  style={"maxWidth": "1000px"}))], no_gutters=True,),
    
        
        #dbc.Toast(Images, id="toggle-add-Images",header="Generated Images",is_open=False,icon="primary",dismissable=True,  style={"maxWidth": "800px"}),
        #dbc.Toast(Settings, id="toggle-add-Settings",header="Edit configuration file",is_open=False,icon="primary",dismissable=True,  style={"maxWidth": "450px"}),
        #dbc.Toast(Result, id="toggle-add-Result",header="Output",is_open=False,icon="primary",dismissable=True),
        dbc.Row([dbc.Col(dbc.Toast(choose_face, id="toggle-add-face",header="Choose face profile",is_open=False,icon="primary",dismissable=True,  style={"maxWidth": "1000px"})),
        dbc.Col(dbc.Toast(html.Div(id = 'daacsx'), is_open=False, icon="primary",dismissable=True,  style={"maxWidth": "1000px"}))], no_gutters=True,),
    
        html.Hr(className="my-2"),
        #html.P("Don't close this window during the process. You can Play or Download the Generated video anytime by clicking on the Result Tab ", id = 'output_text_3'),
      dcc.Interval(
            id='interval-1',
            interval=10000, # in milliseconds
            n_intervals=0
        )
    ,
        dcc.Interval(
            id='interval-2',
            interval=60000, # in milliseconds
            n_intervals=0
        ), 
        
        dcc.Interval(
            id='interval-3',
            interval=1000, # in milliseconds
            n_intervals=0
        ), 
        dcc.Interval(
            id='interval-4',
            interval=500, # in milliseconds
            n_intervals=0
        ),
        
    dbc.Tooltip('Start the process', target="Start-click"),
    #dbc.Tooltip('Show generated results', target="Images-addclick"),
    dbc.Tooltip('Stop training', target="Resetal-addclick"),
    dbc.Tooltip('Delete workspace and model', target="delete-addclick"),
    #dbc.Tooltip('Edit Configuration file', target="Settings-addclick"),
    
    ]
)
upload= loading([(dcc.Upload([
        'Drag and Drop or ',
        html.A('Select a File')
        ], style={
        'width': '100%',
        'height': '100%',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
        
        }, id = 'upload-file')),
        (html.Div(id = 'uploading'))])
        
        
Youtube = loading([dbc.InputGroup(
            [dbc.Input(bs_size="sm", id = 'utube-url'), dbc.Button("Submit", color="primary", id = 'utube-button', size="sm")],
            size="sm",
        ),
    
    (html.Div(id = 'youtube-display'))
])    
#<i class="fab fa-youtube"></i>
#<i class="fas fa-cloud-upload-alt"></i>
#<i class="fas fa-trash-restore"></i>
controls = dbc.Jumbotron(
    [
        html.H1(["Source Video" ]),
        html.Hr(),
        html.P(['Total ',dbc.Badge(video_index(), id = 'n_video', color="light", className="ml-1"), 
        ' videos added of', dbc.Badge(str(duration())+'s', id = 'n_sec_video', color="light",
        className="ml-1"), ' duration'], className="lead",
        ),
        
        dbc.ButtonGroup(
            [dbc.Button(outline=True, id = 'Youtube-addclick',active=False, color="primary", className="fab fa-youtube"),
            dbc.Button(outline=True, id = 'Upload-addclick',active=False, color="primary", className="fas fa-cloud-upload-alt"),
            dbc.Button(
                outline=True, color="danger", disabled = True, active=False,className="fas fa-trash-restore", id = 'Reset-addclick'),
              ],
      
            className="mr-1"),
    
        html.Hr(className="my-2"),
        dbc.Toast(upload, id="toggle-add-upload",header="Upload your Video",is_open=False,icon="primary",dismissable=True, style={"maxWidth": "500px"}),
        dbc.Toast(Youtube, id="toggle-add-utube",header="Download Video from Youtube",is_open=False,icon="primary",dismissable=True,  style={"maxWidth": "500px"}),
        dbc.Toast(record, id="toggle-add-record",header="Record your own Video",is_open=False,icon="primary",dismissable=True,  style={"maxWidth": "500px"}),
        #html.Hr(className="my-2"),
        #html.P("You haven\'t added any videos. Let\'s add one. You have the option to add video by Upload, Youtube or Webcam", id = 'output_text')
        
        dbc.Tooltip('Add videos from Youtube', target="Youtube-addclick"),
        dbc.Tooltip('Upload from your machine', target="Upload-addclick"),
        dbc.Tooltip('Reset', target="Reset-addclick"),
      
    ]
)
upload_= loading([(dcc.Upload([
        'Drag and Drop or ',
        html.A('Select a File')
        ], style={
        'width': '100%',
        'height': '100%',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
        
        }, id = 'upload-file_2')),
        (html.Div(id = 'uploading_2'))])
        
        
        
Youtube_ = loading([dbc.InputGroup(
            [dbc.Input(bs_size="sm", id = 'utube-url_2'), dbc.Button("Submit", color="primary", id = 'utube-button_2', size="sm" )],
            size="sm",
        ),
    
    (html.Div(id = 'youtube-display_2'))
])    
controls_ = dbc.Jumbotron(
    [
        html.H1("Target Video"),
        html.Hr(),
        html.P(['Total ',dbc.Badge(video_index2(), id = 'n_video_2', color="light", className="ml-1"), 
        ' videos added of', dbc.Badge(str(duration())+'s', id = 'n_sec_video_2', color="light",
        className="ml-1"), ' duration'], className="lead",
        ),
        
        dbc.ButtonGroup(
            [dbc.Button(outline=True, id = 'Youtube-addclick_2',active=False, color="primary", className="fab fa-youtube"),
            dbc.Button(outline=True, id = 'Upload-addclick_2',active=False, color="primary", className="fas fa-cloud-upload-alt"),
            dbc.Button(
                outline=True, color="danger", disabled = True, active=False,className="fas fa-trash-restore", id = 'Reset-addclick_2'),
                ],
      
            className="mr-1"),
            
        html.Hr(className="my-2"),
        dbc.Toast(upload_, id="toggle-add-upload_2",header="Upload your Video",is_open=False,icon="primary",dismissable=False,  style={"maxWidth": "500px"}),
        dbc.Toast(Youtube_, id="toggle-add-utube_2",header="Download Video from Youtube",is_open=False,icon="primary",dismissable=False,  style={"maxWidth": "500px"}),
        #dbc.Toast(record, id="toggle-add-record_2",header="Record your own Video",is_open=False,icon="primary",dismissable=False),
        #html.Hr(className="my-2"),
        #html.P("You haven\'t added any videos here. Let\'s add one. You have the option to add video by Upload, Youtube or Webcam", id = 'output_text_2')
        dbc.Tooltip('Add videos from Youtube', target="Youtube-addclick_2"),
        dbc.Tooltip('Upload from your machine', target="Upload-addclick_2"),
        dbc.Tooltip('Reset', target="Reset-addclick_2"),
      
    ]
)
#cc =  dbc.Container([dbc.Row([dbc.Col(controls), dbc.Col(controls)])])
Upload_Tab = dbc.Row(
            [
                dbc.Col(controls),#width={"size": 6, "offset": 3}
                dbc.Col(controls_),               
            ],
            align="center",       
            
        )
        
        
Training_Tab =  dbc.Row(
            [
            
              dbc.Col(controls_start),
                    
                
            ],
            align="center",
              
        )
##########print (len(npy_files))
tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(Upload_Tab, label="Upload", tab_id="tab-1"),
                dbc.Tab(Training_Tab, label="Train/Resume", tab_id="tab-2"),
                # dbc.Tab(Convert_Tab, label="Convert", tab_id="tab-3"),
            ],
            id="tabs",
            active_tab="tab-1",
            
        ),
        html.Div(id="content"),
    ]
)
modal_error = dbc.Modal(
            [
                dbc.ModalHeader("Unexpected Error!"),
                dbc.ModalBody(id = 'modal_error_details'),
                dbc.ModalFooter(
                    html.A(dbc.Button("Refresh", id="Refresh_error"), href='/')
                ),
            ],
            id="modal_error",
        )
        
font_style = {
'font-family': "Comic Sans MS",
'font-size': '31px',
'letter-spacing': '0.4px',
'word-spacing': '2px',
'color': '#000000',
'font-weight': '700',
'text-decoration': 'none',
'font-style':'normal',
'font-variant': 'normal',
'text-transform': 'none',
'height' :'45px',
"text-align":"center"
}

navbar = dbc.NavbarSimple(
    
    style = font_style,
    brand="Dr.face",
    brand_href="#",
    color="light",
   # dark=True,
)        
        
main_panel = dbc.Container(
    [   #html.Br(),
        
        
        #navbar,
        #html.H1([dbc.Badge("Dr.",  pill=True,color ='success',className="ml-1"),'face'],  style=font_style,),
        #html.Br(),
       
        tabs,
        
        modal_error,
        
        
        dbc.Modal(
            [ 
                dbc.ModalHeader("Error! No training data found"),
                dbc.ModalBody("Please restart the program and try again."
                        ),
                        
                 
          
                
            ],
            id="error_modal_no_data",
            #centered=True,
            backdrop = 'static',
            #autoFocus = True
          
        ),
        
        dbc.Modal([
          
                dbc.ModalHeader("GPU not found"),
                dbc.ModalBody("You need a GPU to run the program. If you are already in a GPU machine then make sure your machine has GPU drivers installed."),
                
          
                
            ],
            id="gpu-error",
            #centered=True,
            backdrop = 'static',
            #autoFocus = True
          
        ),
        
        html.Div(id = 'temp1', style = {'display': 'none'})    ,
        html.Div(id = 'temp2', style = {'display': 'none'}),
        html.Div(id = 'temp1_2', style = {'display': 'none'})    ,
        html.Div(id = 'temp2_2', style = {'display': 'none'}),
        html.Div(id = 'tempvar', style = {'display': 'none'}), 
        html.Div(id = 'refresh__', style = {'display': 'none'})   ,
        html.Div(id = 'confirm_delete', style = {'display': 'none'}) ,
        html.Div(id = 'temp_delete', style = {'display': 'none'})    ,   
        html.Div(id = 'temp_4', style = {'display': 'none'})  ,
        html.Div(id = 'start_text_continue_', style = {'display': 'none'})                
],fluid=True, #style = {'width':'60%'}
)




app.layout = dbc.Modal(
            [
                dbc.ModalHeader(dbc.Row(
            [
                dbc.Col(html.Div(html.Img(src = '/assets/logo.svg', style = {'height':'48px'}))),
               # dbc.Col('stop', ),
            ],
            justify="between",
        )),
                dbc.ModalBody(main_panel),
                
            ],
            id="main_panel",
            centered=True,
            is_open = True,
            backdrop= 'static',
            style={"maxWidth": "1000px"},
            scrollable=True,
          
        )

@app.callback([Output('option_import_select','options'), Output('select_train_scratch','options')], [Input('interval-3', 'n_intervals')])        
def toggle_modal(s): 
    global option_import  
    option_import = []
    
    files_ = os.listdir('/data')


    for j,idx in enumerate(files_[::-1]):
        if os.path.isdir(os.path.join('/data',idx,'model')):
            if len(os.listdir(os.path.join('/data',idx,'model')))>4:
                option_import.append({"label": '/data/'+idx , "value" : j+1})
            
    if len(option_import)>0:
        ll = [{'label' : 'Training from scratch', 'value' : '0'}, {'label' : 'Import a model', 'value' : '1'}]
        
    else:
        ll =  [{'label' : 'Training from scratch', 'value' : '0'}, {'label' : 'Import a model', 'value' : '1', 'disabled':True}]
        
    return option_import, ll


@app.callback([Output('option_import_select_div','style'), Output('select_mode_form','style'), Output('select_resolution_form','style')], [Input('select_train_scratch', 'value')])        
def toggle_modal(s): 
    #print (len(GPUs_opts))
    if s == '1':
    
        return {'display':''}, {'display':'none'}, {'display':'none'}
        
    else:
        return  {'display':'none'}, {'display':''},{'display':''}
        
@app.callback(Output('gpu-error','is_open'), [Input('temp_4', 'children')])        
def toggle_modal(s): 
    #print (len(GPUs_opts))
    if len(GPUs_opts)==1:
    
        return True
    else:
        return False
@app.callback(
    Output("toggle-add-upload", "is_open"),
    [Input("Upload-addclick", "n_clicks")], [State("toggle-add-upload", "is_open")]
)
def open_toast2(n, is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    if n:
        return not is_open
    else:
        return  is_open
@app.callback(
    Output("Upload-addclick", "active"),
    [Input("toggle-add-upload", "is_open")]
)
def open_toast2(is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    return is_open
@app.callback(
    Output("toggle-add-utube", "is_open"),
    [Input("Youtube-addclick", "n_clicks")], [State("toggle-add-utube", "is_open")]
)
def open_toast2(n, is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    if n:
        return not is_open
    else:
        return  is_open
@app.callback(
    Output("Youtube-addclick", "active"),
    [Input("toggle-add-utube", "is_open")]
)
def open_toast2(is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    return is_open
    
    
@app.callback(
    [Output("toggle-add-record", "is_open"), Output("Record-addclick", "active")],
    [Input("Record-addclick", "n_clicks")],[State("toggle-add-record", "is_open"), State("Record-addclick", "active")]
)
def open_toast3(n, is_open, is_active):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    if n:
        return not is_open, not is_active
    else:
        return  is_open,  is_active
        
        
        
        
@app.callback(Output('New_modal_','is_open'), [Input('New_workspace', 'n_clicks'), Input('New_modal_Butt', 'n_clicks')], [State("New_modal_", "is_open")],)        
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open        
        
        
@app.callback(Output('Open_modal_','is_open'), [Input('Open_workspace', 'n_clicks'), Input('Open_modal_Butt', 'n_clicks')], [State("Open_modal_", "is_open")],)        
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open        
        
        
@app.callback([Output('drive_dload_','is_open'), Output('drive_dload_loading','children')], [Input('drive_dload', 'n_clicks'), Input('drive_dload_butt', 'n_clicks')], [State("drive_dload_input", "value")],)        
def toggle_modal(n1, n2, n3):
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    if trigger_id == 'drive_dload.n_clicks':
        return True, ''
        
    if trigger_id == 'drive_dload_butt.n_clicks':
        model = [i['label'] for i in option_drive if i['value'] == int(n3)][0]
        filep = os.path.join(drive_path, 'Dr.Face', model)
        print ('\nExtracting files... '+ filep)
        import zipfile
        with zipfile.ZipFile(filep, 'r') as zip_ref:
            zip_ref.extractall('/')
        print ('Completed\n')
        return False, ''
    else:
        return dash.no_update, dash.no_update
          
    
@app.callback(Output('New_workspace','disabled'), [Input('interval-3', 'n_intervals')])       
def toggle_modal(n):
    if len(tar_vids_clip)>0 and len(src_vids_clip)>0 :
        return False
    else:
        return True
        
@app.callback(Output('start_text_input_','options'), [Input('interval-1', 'n_intervals')])       
def toggle_modal(n):
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    if trigger_id == 'interval-1.n_intervals':
        
        
        trigger_id = dash.callback_context.triggered[0]['prop_id']
        error_modal_no_data = dash.no_update
        files =os.listdir('/data')
        files.sort(key=lambda x: os.path.getmtime('/data/'+x))
        option_ = []#[{"label": '(1) New Wokspace', "value" : 1}, {"label": '(2) Resume Workspace', "value" : 1}, {"label": '(3) Load Workspace', "value" : 2, 'disabled': True}]
        for j,idx in enumerate(files[::-1]):
            option_.append({"label": idx , "value" : j+2})
            
        
        return [{"label":'/data/'+i["label"]  + ' - modified '+get_timeago('/data/'+i["label"] ), "value":i["value"]} for i in option_]
    
    else:
        return dash.no_update
        
        
        
                        
@app.callback( [Output('start_text_input', 'value'), Output('face_type_select', 'value'), Output('start_text_continue_', 'children'), Output('error_modal_no_data', 'is_open'),],
[Input('New_modal_Butt', 'n_clicks'), Input('Open_modal_Butt', 'n_clicks'), ],
[State('select_mode', 'value'),State('select_resolution', 'value'),State('select_device_', 'value'), State('select_device', 'value'), State('select_Batchsize_', 'value'), State('select_Batchsize', 'value'), State('start_text_input_','value'), State('start_text_new','value'), State('option_import_select','value'), State('select_train_scratch','value')] )     
        
def update(n1,n2,select_mode,select_resolution, select_device_, select_device, select_Batchsize_,select_Batchsize,start_text_input_, start_text_new, option_import_select, select_train_scratch):
    global option_
    global counter_children
    
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    
    if trigger_id=='New_modal_Butt.n_clicks' or  trigger_id=='Open_modal_Butt.n_clicks': print ('\nStarting... \n')
    error_modal_no_data = dash.no_update
    
    files =os.listdir('/data')
    files.sort(key=lambda x: os.path.getmtime('/data/'+x))
    
    option_ = []#[{"label": '(1) New Wokspace', "value" : 1}, {"label": '(2) Resume Workspace', "value" : 1}, {"label": '(3) Load Workspace', "value" : 2, 'disabled': True}]
    for j,idx in enumerate(files[::-1]):
        option_.append({"label": idx , "value" : j+2})
        
    option__1 = [{"label":'/data/'+i["label"]  + ' - modified '+get_timeago('/data/'+i["label"] ), "value":i["value"]} for i in option_]
        
    if trigger_id == 'New_modal_Butt.n_clicks':
        if start_text_new == None or start_text_new.strip() == '': start_text_new = 'Untitled'
        #print (start_text_new)
        f = open('/tmp/model.txt','w+')
        
        convert_id = f.write('_'.join(start_text_new.split(' ')))
        f.close()
        #print (datadir()+'/')
        if os.path.isdir(datadir()+'/'):shutil.rmtree(datadir()+'/')
        
        if not os.path.isdir(datadir()+''): os.mkdir(datadir()+'')
        if not os.path.isdir(datadir()+'/data_dst'): os.mkdir(datadir()+'/data_dst')
        if not os.path.isdir(datadir()+'/data_src'): os.mkdir(datadir()+'/data_src')
        if not os.path.isdir(datadir()+'/model'): os.mkdir(datadir()+'/model')
        if not os.path.isdir(datadir()+'/preview'): os.mkdir(datadir()+'/preview')
        #if not os.path.isdir(datadir()+'/params'): os.mkdir(datadir()+'/params')
        for filename in glob.glob("assets/*.mp4"):os.remove(filename)
        if not os.path.isdir('assets'): os.mkdir('assets')
        if type(select_device) == list: select_device = ','.join([str(i) for i in select_device])
        #print (select_device)
        #print ('asknksafnklkl')
        
        if select_train_scratch == '1':
        
            files_ = os.listdir('/data')
        
            option_import = []
            for j,idx in enumerate(files_[::-1]):
                if os.path.isdir(os.path.join('/data',idx,'model')):
                    if len(os.listdir(os.path.join('/data',idx,'model')))>4:
                        option_import.append({"label": '/data/'+idx , "value" : j+1})
        
            model_pretrained = [i['label'] for i in option_import if i['value'] == int(option_import_select)][0]
            #print (model_pretrained)
            
            f = open(os.path.join(model_pretrained, '.params'), 'r')
            params_ = {i[:-1].split(' ')[0]:i[:-1].split(' ')[1] for i in f.readlines()}
            f.close()
            def copy_tree_(a,b):
                from distutils.dir_util import copy_tree
                a_i = [i for i in os.listdir(a) if not i.endswith('.jpg')]
                for i in a_i:
                    try:
                        copyfile(os.path.join(a,i), os.path.join(b,i))
                    except:
                        copy_tree(os.path.join(a,i), os.path.join(b,i))
            #copy_tree(os.path.join(model_pretrained, 'model'), datadir()+'/model')
            thr8 = Process(target = copy_tree_, args=(os.path.join(model_pretrained, 'model'), datadir()+'/model',))
            thr8.daemon = True
            thr8.start()
            
            
            
        f = open(datadir()+'/.params', 'w+')
        
        
        
        #print ('sknsfknfesnkenf')
        if select_train_scratch == '1':
        
            f.write('facetype '+str(params_['facetype'])+'\n')
            f.write('Quality '+str(params_['Quality'])+'\n')
            
        else:
            f.write('facetype '+str(select_mode)+'\n')
            f.write('Quality '+str(select_resolution)+'\n')
            
        f.write('device '+str(select_device)+'\n')
        f.write('Batchsize '+str(select_Batchsize)+'\n')
        f.write('suggest_batch_size '+str(0)+'\n')
        f.close()
        
        counter_children = counter_children + 1
        
        open('/tmp/start','w+').close()
        return 1, '1', counter_children, error_modal_no_data
        
    elif trigger_id == 'Open_modal_Butt.n_clicks':
    
        model = [i['label'] for i in option_ if i['value'] == int(start_text_input_)][0]
        #print (model)
        f = open('/tmp/model.txt','w+')
        
        convert_id = f.write(model)
        f.close()
        
        if os.path.isfile(datadir()+"/data_dst.mp4") and os.path.isfile(datadir()+"/data_src.mp4"):
    
            if not os.path.isdir(datadir()+''): os.mkdir(datadir()+'')
            if not os.path.isdir(datadir()+'/data_dst'): os.mkdir(datadir()+'/data_dst')
            if not os.path.isdir(datadir()+'/data_src'): os.mkdir(datadir()+'/data_src')
            if not os.path.isdir(datadir()+'/model'): os.mkdir(datadir()+'/model')
            if not os.path.isdir(datadir()+'/preview'): os.mkdir(datadir()+'/preview')
            for filename in glob.glob("assets/*.mp4"):os.remove(filename)
            if not os.path.isdir('assets'): os.mkdir('assets')
            
            
            if type(select_device_) == list: select_device_ = ','.join([str(i) for i in select_device_])
            f = open(datadir()+'/.params', 'r')
            params = {i[:-1].split(' ')[0]:i[:-1].split(' ')[1] for i in f.readlines()}
            params['Batchsize'] = select_Batchsize_
            params['device'] = select_device_
            #print ('asknksafnklkl')
            ##print (params['device'])
            #print ('asknksafnklkl')
            f.close()
            #print ('asknksafnklkl')
            f = open(datadir()+'/.params', 'w+')
            f.write('facetype '+str(params['facetype'])+'\n')
            f.write('Quality '+str(params['Quality'])+'\n')
            f.write('device '+str(params['device'])+'\n')
            f.write('Batchsize '+str(params['Batchsize'])+'\n')
            f.write('suggest_batch_size '+str(params['suggest_batch_size'])+'\n')
            f.close()
            #print ('trigerr')
            #print (start_text_input_)
            counter_children = counter_children + 1
            #print (counter_children)
            open('/tmp/start','w+').close()
            return start_text_input_, dash.no_update, counter_children, error_modal_no_data
            
        else:
                
            return dash.no_update, dash.no_update,  dash.no_update, True
    else:
            return  dash.no_update, dash.no_update, dash.no_update,error_modal_no_data
 
 
 
'''
New_modal = html.Div([html.Br(),
dbc.Input(placeholder="Enter Workspace Name", type="start_text_new"),
html.Br(),
dbc.FormGroup([
 
        dbc.Label("Select Mode:"),
        dbc.RadioItems(
            options=[{'label' : 'Face & Head', 'value' : '0'}, {'label' : 'Full face', 'value' : '1'}, {'label' : 'Half Face', 'value' : '2'}],
            value='1',
            id="select_mode",
            inline=True,
        ),]),
        
 dbc.FormGroup([
 
        dbc.Label("Select Quality:"),
        dbc.RadioItems(
            options=[
                {"label": "High", "value": 1},
                {"label": "Medium", "value": 2},
                {"label": "Low", "value": 3},
            ],
            value=1,
            id="select_resolution",
            inline=True,
        ),]
    
),
 dbc.FormGroup(
    [
        dbc.Label("Select Device:"),
        dbc.Checklist(
            options=GPUs_opts,
            value=GPUs_opts[0]['value'],
            id="select_device",
            inline=True,
        ),
    ]
),
 
 
 
 
 
 
 
 
 
 
 '''
 
 
 
 
 
 
        
        
@server.route('/video_feed')
def video_feed():
    global camera 
    camera = VideoCamera()
    if camera.open:
        return Response(gen(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
@app.callback(
    [Output('rec_button', 'children'),Output("Record-addclick", "n_clicks")],
    [Input('rec_button', 'n_clicks')],
    [State('rec_button', 'children')])
def update_button(n_clicks, butt):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    global camera
    
    if n_clicks is not None:
        
        if n_clicks%3==1:
            camera.start()
            return 'Stop', 1
        elif n_clicks%3==2:
            camera.stop()
            return 'Add', 1
        elif n_clicks%3==0:
          
        
            copyfile('videos/Source/Record/temp.mp4', 'videos/Source/final/temp'+str(video_index())+'.mp4')
            return 'Added Successfully', 2
        
    else:
        return butt, 0
    
    
@app.callback(
    Output('uploading', 'children'),
    [Input('upload-file', 'contents')])
def update_upload(data):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    if data is not None:
        content_type, content_string = data.split(',')
        decoded = base64.b64decode(content_string)
        ############print (decoded)
        with open('videos/Source/Upload/temp.mp4', "wb") as fp:
            fp.write(decoded)
        global src_vids
        global HEIGHT
        VID = VideoFileClip('videos/Source/Upload/temp.mp4')
        #VID = VID.resize((int((VID.aspect_ratio*HEIGHT)//2)*2, HEIGHT))
        src_vids.append(VID)
        frame = VID.get_frame(0)
        frame = imutils.resize(frame, height=64)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        frame = base64.b64encode(frame)
        return html.Div( 
            [html.Hr(), html.Img(id = 'playback', style={
            'width': '100%',
            'height': '100%', 'padding-left':'8.5%', 'padding-right':'8.5%'
            }, src = 'data:image/png;base64,{}'.format(frame.decode())), dcc.RangeSlider(
                id='my-range-slider',
                min=0,
                max=1000,
                step=1,
                value=[1, 999], marks = {0: '0:00', 1000: get_sec2time(VID.duration)}),
                  html.Div(dbc.Button(["+",  dbc.Badge(str(int(VID.duration)), id = 'n_upload', color="primary", className="ml-1")], id='crop_button', color="light", size="sm",  style = {'margin-top': '-20px', 'font-weight': 'bold'}),style = {'text-align':'center'})])
  
    
@app.callback(
    Output('youtube-display', 'children'),
    [Input('utube-button', 'n_clicks')],[State('utube-url', 'value')])
def update_youtube(n, url):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    
    if n is not None:
        ytdl_format_options = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'videos/Source/Youtube/temp'
            
        }
        
        files = glob.glob('videos/Source/Youtube/temp*')
        if len(files)>0:
            for i in files:
                os.remove(i)
        
        for _ in range(5):
            try:
                with youtube_dl.YoutubeDL(ytdl_format_options) as ydl:
                    ydl.download([url])
                break
            except:
                pass
        
        global src_vids
        global HEIGHT
        VID = VideoFileClip('videos/Source/Youtube/temp.mp4')
        #VID = VID.resize((int((VID.aspect_ratio*HEIGHT)//2)*2, HEIGHT))
        src_vids.append(VID)
        frame = VID.get_frame(0)
        frame = imutils.resize(frame, height=64)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        frame = base64.b64encode(frame)
        
  
        
        return html.Div( 
            [html.Hr(), html.Img(id = 'playback_utube', style={
            'width': '100%',
            'height': '100%','padding-left':'8.5%', 'padding-right':'8.5%'
            }, src = 'data:image/png;base64,{}'.format(frame.decode())), dcc.RangeSlider(
                id='my-range-slider_utube',
                min=0,
                max=1000,
                step=1,
                value=[1, 999], marks = {0: '0:00', 1000: get_sec2time(VID.duration)}), 
                html.Div(dbc.Button(['+', dbc.Badge(str(int((VID.duration))), id = 'n_utube', color="primary", className="ml-1")],id='crop_button_utube', 
color="light", size="sm",  style = {'margin-top': '-20px','font-weight': 'bold'}), style = {'text-align':'center'})])
@app.callback(
    [
      
      Output("Reset-addclick", "disabled"),
      Output("n_video", "children"),
      Output("n_sec_video", "children"),
      Output("temp_delete", "children")],
    [Input('temp1', 'children'), 
      Input('temp2', 'children'),
      Input('Reset-addclick', 'n_clicks'),
      Input('Resetal-addclick', 'n_clicks'),
    ],
      [
      State("Reset-addclick", "disabled"),
      State("n_video", "children"),
      State("n_sec_video", "children")]
      )
def update_details(t1, t2, n, n1, s2, s3, s4):
  ###print'######################################################')
  ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
  ###print'######################################################')
  trigger_id = dash.callback_context.triggered[0]['prop_id']
  trgger_value = dash.callback_context.triggered[0]['value']
  global src_vids_clip
    
  global src_vids
    
  global tar_vids_clip
    
  global tar_vids
  
  
  if trigger_id == 'Resetal-addclick.n_clicks':
    
    
    src_vids_clip = []
    
    
    tar_vids_clip = []
    

   
    def sfv():
        for i in thread_list:
            os.system('kill -9 '+str(i.pid)+'> /dev/null 2>&1')
            #i.terminate()

        killall()
        for filename in glob.glob("/tmp/*npy"):os.remove(filename)
        if os.path.isdir(datadir()+'/data_dst/merged'):
            shutil.rmtree(datadir()+'/data_dst/merged')
            os.mkdir (datadir()+'/data_dst/merged')
        #shutdown() 
        if os.path.isfile('/tmp/running'): os.remove('/tmp/running')
        time.sleep(1)
    
    thr5 = Process(target = sfv, args=())
    thr5.daemon = True
    thr5.start()


        
    return  [ True, str(video_index()), str(duration()) + 's', ' ']
  
  elif trigger_id == 'Reset-addclick.n_clicks':
  
    src_vids_clip = []
    #src_vids = []
    
    #shutil.rmtree('videos/Source/Final'); os.mkdir('videos/Source/Final')
    output = 'You have added total ' + str(video_index()) + ' video(s). You can add more videos' 
    return  [True, str(video_index()), str(duration()) + 's',  dash.no_update]
  elif t1 == 'True' or t2 == 'True':
    output = 'You have added total ' + str(video_index()) + ' video(s). You can add more videos' 
    ###########print ('ffff')
    return [ False, str(video_index()), str(duration()) + 's', dash.no_update]
  else:
    return [s2, s3, s4, dash.no_update]
@app.callback(
    [Output('playback_utube', 'src'),
      #Output("Youtube-addclick", "n_clicks"), 
      Output("temp1", "children"),
      Output("n_utube", "children"),
      Output("my-range-slider_utube", "marks")],
    [Input('my-range-slider_utube', 'value'), 
      Input('crop_button_utube', 'n_clicks') 
      ],[State('playback_utube', 'src')])
def upload_playback_utube(rang, n_clicks, s):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    global src_vids
    global src_vids_clip
    
    
  
    VID = src_vids[-1]
    
    #cap = cv2.VideoCapture(file)
    fps = VID.fps 
    T = VID.duration
    #fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = T*fps
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    trgger_value = dash.callback_context.triggered[0]['value']
  
    if trigger_id == 'crop_button_utube.n_clicks':
        
        
    
        ###########print (n_clicks)
        
        #res, frame = cap.read()
        #frame = cv2.resize(frame, (100, 70),interpolation=cv2.INTER_CUBIC)
        #ret, frame = cv2.imencode('.png', frame)
        #frame = base64.b64encode(frame)
        ############print (rang)
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        VID = VID.subclip(str_time, end_time)
        #del src_vids[-1]
        src_vids_clip.append(VID)
      
        
        #cap.release()
        output = 'You have added total ' + str(video_index()) + ' video(s). You can add more videos.' 
        length = VID.duration
        ###########print ('jkbdasflsfkafbkasbkfasaskasksbkabkaj' )
        ###########print (length)
        return [s, 'True', str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
        
    else:
        
        global slider_prev_instance 
    
  
        ###########print (totalNoFrames)
        if slider_prev_instance[0] == rang[0]:
            time_n = int(T*rang[1]/1000)
        elif slider_prev_instance[1] == rang[1]:
            time_n = int(T*rang[0]/1000)
        else:
            time_n = int(T*rang[0]/1000)
        slider_prev_instance = rang
        #cap.set(1, frame_number)
        #res, frame = cap.read()
        frame = VID.get_frame(time_n)
        frame = imutils.resize(frame, height=64)
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        #frame = cv2.resize(frame, (100, 70),interpolation=cv2.INTER_CUBIC)
        ############print (res)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        #frame = cv2.resize(frame, (128,128))
        length = end_time - str_time
        frame = base64.b64encode(frame)
        
        return ['data:image/png;base64,{}'.format(frame.decode()),'False', str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
@app.callback(
    [Output('playback', 'src'), 
      #Output("Upload-addclick", "n_clicks"), 
      Output("temp2", "children"),
      Output("n_upload", "children"),
      Output("my-range-slider", "marks")],
    [Input('my-range-slider', 'value'), Input('crop_button', 'n_clicks')],[State('playback', 'src')])
def upload_playback(rang,n_clicks,s):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    global src_vids
    global src_vids_clip
    
    
  
    VID = src_vids[-1]
    
    #cap = cv2.VideoCapture(file)
    fps = VID.fps 
    T = VID.duration
    #fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = T*fps
    
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    trgger_value = dash.callback_context.triggered[0]['value']
  
    if trigger_id == 'crop_button.n_clicks':
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        VID = VID.subclip(str_time, end_time)
        src_vids_clip.append(VID)
        length = VID.duration
        output = 'You have added total ' + str(video_index()) + ' video(s). You can add more videos' 
    
        return [s, 'True',str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
        
    else:
        
        global slider_prev_instance 
    
  
        ###########print (totalNoFrames)
        if slider_prev_instance[0] == rang[0]:
            time_n = int(T*rang[1]/1000)
        elif slider_prev_instance[1] == rang[1]:
            time_n = int(T*rang[0]/1000)
        else:
            time_n = int(T*rang[0]/1000)
        slider_prev_instance = rang
        frame = VID.get_frame(time_n)
        frame = imutils.resize(frame, height=64)
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        #frame = cv2.resize(frame, (100, 70),interpolation=cv2.INTER_CUBIC)
        ############print (res)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        frame = base64.b64encode(frame)
        length = end_time - str_time
        return ['data:image/png;base64,{}'.format(frame.decode()), 'False', str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
@app.callback(
    Output("toggle-add-upload_2", "is_open"),
    [Input("Upload-addclick_2", "n_clicks")], [State("toggle-add-upload_2", "is_open")]
)
def open_toast2(n, is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    if n:
        return not is_open
    else:
        return  is_open
@app.callback(
    Output("Upload-addclick_2", "active"),
    [Input("toggle-add-upload_2", "is_open")]
)
def open_toast2(is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    return is_open
@app.callback(
    Output("toggle-add-utube_2", "is_open"),
    [Input("Youtube-addclick_2", "n_clicks")], [State("toggle-add-utube_2", "is_open")]
)
def open_toast2(n, is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    if n:
        return not is_open
    else:
        return  is_open
@app.callback(
    Output("Youtube-addclick_2", "active"),
    [Input("toggle-add-utube_2", "is_open")]
)
def open_toast2(is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    return is_open
    
@app.callback(
    [Output("toggle-add-record_2", "is_open"), Output("Record-addclick_2", "active")],
    [Input("Record-addclick_2", "n_clicks")],[State("toggle-add-record_2", "is_open"), State("Record-addclick_2", "active")]
)
def open_toast3(n, is_open, is_active):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    if n:
        return not is_open, not is_active
    else:
        return  is_open,  is_active
@server.route('/video_feed_')
def video_feed_():
    global camera 
    camera = VideoCamera()
    if camera.open:
        return Response(gen(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
@app.callback(
    [Output('rec_button_2', 'children'),Output("Record-addclick_2", "n_clicks")],
    [Input('rec_button_2', 'n_clicks')],
    [State('rec_button_2', 'children')])
def update_button(n_clicks, butt):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    global camera
    
    if n_clicks is not None:
        
        if n_clicks%3==1:
            camera.start()
            return 'Stop', 1
        elif n_clicks%3==2:
            camera.stop()
            return 'Add', 1
        elif n_clicks%3==0:
      
            copyfile('videos/Target/Record/temp.mp4', 'videos/Target/final/temp'+str(video_index2())+'.mp4')
            return 'Added Successfully', 2
        
    else:
        return butt, 0
    
@app.callback(
    [
      
      Output("Reset-addclick_2", "disabled"),
      Output("n_video_2", "children"),
      Output("n_sec_video_2", "children")],
    [Input('temp1_2', 'children'), 
      Input('temp2_2', 'children'),
      Input('Reset-addclick_2', 'n_clicks'),Input('Resetal-addclick', 'n_clicks'),],
    
      [
      State("Reset-addclick_2", "disabled"),
      State("n_video_2", "children"),
      State("n_sec_video_2", "children")]
      )
def update_details(t1, t2, n, ss, s2, s3, s4):
  ###print'######################################################')
  ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
  ###print'######################################################')
  trigger_id = dash.callback_context.triggered[0]['prop_id']
  trgger_value = dash.callback_context.triggered[0]['value']
  #if trigger_id == 'Resetal-addclick.n_clicks': print ('abc3')
  if trigger_id == 'Reset-addclick_2.n_clicks':
    
    #global tar_vids
    #tar_vids = []
    global tar_vids_clip
    tar_vids_clip = []
    #output = 'You have added total ' + str(video_index2()) + ' video(s). You can add more videos' 
    return  [True, str(video_index2()), str(duration2()) + 's']
  elif t1 == 'True' or t2 == 'True':
    #output = 'You have added total ' + str(video_index2()) + ' video(s). You can add more videos' 
    ###########print ('ffff')
    return [ False, str(video_index2()), str(duration2()) + 's']
  elif trigger_id == 'Resetal-addclick.n_clicks':
  
    return [ dash.no_update, str(video_index2()), str(duration2()) + 's']
  else:
    return [s2, s3, s4]
    
@app.callback(
    Output('uploading_2', 'children'),
    [Input('upload-file_2', 'contents')])
def update_upload(data):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    if data is not None:
        content_type, content_string = data.split(',')
        decoded = base64.b64decode(content_string)
        ############print (decoded)
        with open('videos/Target/Upload/temp.mp4', "wb") as fp:
            fp.write(decoded)
            
        global tar_vids
        global HEIGHT
        VID = VideoFileClip('videos/Target/Upload/temp.mp4')
        #VID = VID.resize((int((VID.aspect_ratio*HEIGHT)//2)*2, HEIGHT))
        tar_vids.append(VID)
        frame = VID.get_frame(0)
        frame = imutils.resize(frame, height=64)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        frame = base64.b64encode(frame)
        
        return html.Div( 
            [html.Hr(), html.Img(id = 'playback_2', style={
            'width': '100%',
            'height': '100%', 'padding-left':'8.5%', 'padding-right':'8.5%'
            }, src = 'data:image/png;base64,{}'.format(frame.decode())), dcc.RangeSlider(
                id='my-range-slider_2',
                min=0,
                max=1000,
                step=1,
                value=[1, 999],marks = {0: '0:00', 1000: get_sec2time(VID.duration)}),  html.Div(dbc.Button(['+', dbc.Badge(str(int((VID.duration))), id = 'n_upload_2', color="primary", className="ml-1")], id ='crop_button_2',
                                                                                          color="light", size="sm",  style = {'margin-top': '-20px',  'font-weight': 'bold'}),style = {'text-align':'center'})])
  
    
@app.callback(
    Output('youtube-display_2', 'children'),
    [Input('utube-button_2', 'n_clicks')],[State('utube-url_2', 'value')])
def update_youtube(n, url):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    
    if n is not None:
        ytdl_format_options = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'videos/Target/Youtube/temp'
            
        }
        
        files = glob.glob('videos/Target/Youtube/temp*')
        if len(files)>0:
            for i in files:
                os.remove(i)
        
        
        for _ in range(5):
            try:
                with youtube_dl.YoutubeDL(ytdl_format_options) as ydl:
                    ydl.download([url])
                break
            except:
                pass
                
              
        global tar_vids
        global HEIGHT
        VID = VideoFileClip('videos/Target/Youtube/temp.mp4')
        #VID = VID.resize((int((VID.aspect_ratio*HEIGHT)//2)*2, HEIGHT))
        tar_vids.append(VID)
        
        frame = VID.get_frame(0)
        frame = imutils.resize(frame, height=64)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        frame = base64.b64encode(frame)
        
        return html.Div( 
            [html.Hr(), html.Img(id = 'playback_utube_2', style={
            'width': '100%',
            'height': '100%','padding-left':'8.5%', 'padding-right':'8.5%'
            }, src = 'data:image/png;base64,{}'.format(frame.decode())), dcc.RangeSlider(
                id='my-range-slider_utube_2',
                min=0,
                max=1000,
                step=1,
                value=[1, 999], marks = {0: '0:00', 1000: get_sec2time(VID.duration)}), html.Div(dbc.Button(["+", dbc.Badge(str(int((VID.duration))), id = 'n_utube_2', color="primary", className="ml-1")], id = 'crop_button_utube_2',
                                                                                          color="light", size="sm",  style = {'margin-top': '-20px', 'font-weight': 'bold'}), style = {'text-align':'center'})])
    
@app.callback(
    [Output('playback_utube_2', 'src'),
      #Output("Youtube-addclick_2", "n_clicks"), 
      Output("temp1_2", "children"),
      Output("n_utube_2", "children"),
      Output("my-range-slider_utube_2", "marks")],
    [Input('my-range-slider_utube_2', 'value'), 
      Input('crop_button_utube_2', 'n_clicks')]
      ,[State('playback_utube_2', 'src')])
def upload_playback_utube(rang, n_clicks, s):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    global tar_vids
    global tar_vids_clip
    
    
  
    VID = tar_vids[-1]
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    
    ###########print ('#############################################################################################3')
    ###########print (trigger_id)
    trgger_value = dash.callback_context.triggered[0]['value']
    fps = VID.fps 
    T = VID.duration
    totalNoFrames = T*fps
  
    if trigger_id == 'crop_button_utube_2.n_clicks':
    
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        VID = VID.subclip(str_time, end_time)
        tar_vids_clip.append(VID)
      
    
        output = 'You have added total ' + str(video_index2()) + ' video(s). You can add more videos' 
        length = VID.duration
        
        return [s, 'True', str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
        
    else:
        
        global slider_prev_instance2 
    
  
        ###########print (totalNoFrames)
        if slider_prev_instance2[0] == rang[0]:
            time_n = int(T*rang[1]/1000)
        elif slider_prev_instance2[1] == rang[1]:
            time_n = int(T*rang[0]/1000)
        else:
            time_n = int(T*rang[0]/1000)
        slider_prev_instance2 = rang
        
        frame = VID.get_frame(time_n)
        frame = imutils.resize(frame, height=64)
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.imencode('.png', frame)
        #frame = cv2.resize(frame, (128,128))
        length = end_time - str_time
        frame = base64.b64encode(frame)
        
        return ['data:image/png;base64,{}'.format(frame.decode()), 'False', str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
@app.callback(
    [Output('playback_2', 'src'), 
      #Output("Upload-addclick_2", "n_clicks"), 
      Output("temp2_2", "children"),
      Output("n_upload_2", "children"),
      Output("my-range-slider_2", "marks")],
    [Input('my-range-slider_2', 'value'), Input('crop_button_2', 'n_clicks')],[State('playback_2', 'src')])
def upload_playback(rang,n_clicks,s):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    global tar_vids
    global tar_vids_clip
    
    
  
    VID = tar_vids[-1]
    fps = VID.fps 
    T = VID.duration
    #fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = T*fps
    
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    trgger_value = dash.callback_context.triggered[0]['value']
  
    if trigger_id == 'crop_button_2.n_clicks':
    
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        VID = VID.subclip(str_time, end_time)
        #del src_vids[-1]
        tar_vids_clip.append(VID)
        
        length = VID.duration
        output = 'You have added total ' + str(video_index2()) + ' video(s). You can add more videos' 
    
        return [s,  'True',str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
        
    else:
        
        global slider_prev_instance 
    
  
        ###########print (totalNoFrames)
        if slider_prev_instance[0] == rang[0]:
            time_n = int(T*rang[1]/1000)
        elif slider_prev_instance[1] == rang[1]:
            time_n = int(T*rang[0]/1000)
        else:
            time_n = int(T*rang[0]/1000)
        slider_prev_instance = rang
        frame = VID.get_frame(time_n)
        frame = imutils.resize(frame, height=64)
        str_time = T*rang[0]/1000
        end_time = T*rang[1]/1000
        #frame = cv2.resize(frame, (100, 70),interpolation=cv2.INTER_CUBIC)
        ############print (res)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        ret, frame = cv2.imencode('.png', frame)
        frame = base64.b64encode(frame)
        length = end_time - str_time
        return ['data:image/png;base64,{}'.format(frame.decode()),  'False', str(int((length))) + 's', {0: get_sec2time(str_time), 1000: get_sec2time(end_time)}]
@app.callback([Output("preview_imgs", "children"), Output("preview_graph", "children")],
              
    [Input('interval-2', 'n_intervals'),Input('delete-addclick', 'n_clicks'),Input('preview_divs', 'style'), Input('Progress_modal_',"is_open"),])
def update_images(ints,kd,ff,dkkd):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    
   
    
    
    
    if trigger_id == 'delete-addclick.n_clicks':
    
        killall()
        
        ##global thread_list
    
       
        for i in thread_list:
            os.system('kill -9 '+str(i.pid)) 
                
        if os.path.isdir(datadir()+'/'):
            shutil.rmtree(datadir()+'/')
            
        
        for filename in glob.glob("assets/*.mp4"):os.remove(filename)
        
        
        if not os.path.isdir(datadir()+''): os.mkdir(datadir()+'')
        if not os.path.isdir(datadir()+'/data_dst'): os.mkdir(datadir()+'/data_dst')
        if not os.path.isdir(datadir()+'/data_src'): os.mkdir(datadir()+'/data_src')
        if not os.path.isdir(datadir()+'/model'): os.mkdir(datadir()+'/model')
        if not os.path.isdir('assets'): os.mkdir('assets')
        
        [os.remove(os.path.join('/tmp',i)) for i in os.listdir('/tmp') if i.endswith('.npy')]
        if os.path.isfile('/tmp/running'): os.remove('/tmp/running')
        
        #if not os.path.isfile('/tmp/model.txt'):
    
        #convert_id = (''.join(map(choice,["bcdfghjklmnpqrstvwxz","aeiouy"]*3)))
        #        
        #f = open('/tmp/model.txt','w+')
        #f.write(convert_id)
        #f.close()
        
        time.sleep(1)
    
    
        return ['', '']
        
    elif os.path.isfile('/tmp/model.txt'):
        #print (os.path.isfile('/tmp/model.txt'))
    
        jpgs = glob.glob(datadir()+'/model/*.jpg')
        
        ###########print (jpgs)
        
        if len(jpgs)>1:
            
            img1 = cv2.imread(datadir()+'/model/new_SAEHD_preview_SAEHD.jpg')
            img2 = cv2.imread(datadir()+'/model/new_SAEHD_preview_SAEHD masked.jpg')
            import settings as st
            st = reload(st)
            faceres = st.Resolution
            #print (faceres)
            img3 = np.concatenate([img1[100:100+faceres, 2*faceres:3*faceres], img2[100:100+faceres, 2*faceres:3*faceres], img2[100:100+faceres, 4*faceres:5*faceres], img1[100:100+faceres, 4*faceres:5*faceres]],1)
            #print (img3.shape)
            #print (min(st.Resolution,128))
            img3 = imutils.resize(img3, height = min(st.Resolution,128))
            #print (img3.shape)
            
            
            
            ret, img3 = cv2.imencode('.jpg', img3)
            
            #p = cv2.imread(datadir()+'/model/new_SAEHD_preview_SAEHD masked.jpg')
            #p1 = cv2.imread(datadir()+'/model/new_SAEHD_preview_SAEHD.jpg')
            img3 = base64.b64encode(img3)
            src3 = 'data:image/jpg;base64,{}'.format(img3.decode())
            
            img4 = cv2.resize(255 - img1[:100], (1280,350))
            ret, img4 = cv2.imencode('.jpg', img4)
            
            #p = cv2.imread(datadir()+'/model/new_SAEHD_preview_SAEHD masked.jpg')
            #p1 = cv2.imread(datadir()+'/model/new_SAEHD_preview_SAEHD.jpg')
            img4 = base64.b64encode(img4)
            src4 = 'data:image/jpg;base64,{}'.format(img4.decode())
            del st
            return  [html.Img(id = 'Mask', src = src3, style = {'width' : '100%', 'height' : '100%'}), 
            html.Img(id = 'Mask_1', src = src4, style = {'width' : '100%', 'height' : '100%'})]
            
        
        else:
        
            return ['', '']
    else:
        return dash.no_update,dash.no_update
    
    
@app.callback(
    Output("Start-click", "active"),
    [Input("toggle-add-Progress", "is_open")]
)
def open_toast2(is_open):
    ###########print ('utubessssff')
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    return is_open
    
@app.callback(
    Output("toggle-add-right_frame", "is_open"),
    [Input("interval-3", "n_intervals"), Input('Resetal-addclick', 'n_clicks'), Input('delete-addclick', 'n_clicks'), Input('New_modal_Butt', 'n_clicks'), Input('Open_modal_Butt', 'n_clicks'),]
)
def open_toast1(n,dd,dlld, kll,loo):
    
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    #if trigger_id == 'Resetal-addclick.n_clicks': print ('abc1')
    #if trigger_id == 'New_modal_Butt.n_clicks' or trigger_id == 'Open_modal_Butt.n_clicks':
     #   if os.path.isfile('/tmp/converting'): 
      #      os.remove('/tmp/converting')
       #     return False
    
    #if os.path.isfile('/tmp/converting'):
     #   return True
    
    if os.path.isfile('/tmp/running') and len(glob.glob('assets/*mp4'))>0 and len(glob.glob("/tmp/*npy"))>0:
        return True
        
    else:
    
        return False
@app.callback(
    [Output("status_tooltip", "children"),Output("status_tooltip", "style")],
    [Input("interval-3", "n_intervals")]
)
def open_toast1(n):
    if os.path.isfile('/tmp/running'):
        f = open(datadir()+'/.params', 'r')
        params = {i[:-1].split(' ')[0]:i[:-1].split(' ')[1] for i in f.readlines()}
        f.close()
        fd = {'0': 'Face & Head', '1':'Full face', '2': 'Half Face'}
        Qd = {'1':"High",'2':"Medium",'3':"Low",'4':"Very Low"}
        Rd = {'1':"640",'2':"256",'3':"128",'4':"64"}
        IRd = {'1':"1080",'2':"512",'3':"480",'4':"256"}
        Bd = {'1':'Auto','2':'2','4':'4','8':'8','16':'16'}
        divs= params['device'].split(',') 
        if 'C' in divs: divs.remove('C')
        if len(divs) == 0: device_ = 'CPU'
        else: 
            device_ = ''
            for i in divs: 
                device_ = device_ + 'GPU:'+i + ' '
        
        return [html.Div('Path: '+datadir()),
        html.Div('Mode: '+fd[params['facetype']]),
        html.Div('Quality: '+Qd[params['Quality']]),
        html.Div('Face size: '+Rd[params['Quality']]+'x'+Rd[params['Quality']]),
        html.Div('Image size: '+IRd[params['Quality']]+'x'+IRd[params['Quality']]),
        html.Div('Batchsize: '+Bd[params['Batchsize']]),
        html.Div('Device: '+device_)], {'display':''}
    else:
        return '',{'display':'none'}
    
        
        
@app.callback(Output("tempvar", "value"), [Input('Start-click', 'n_clicks')])
def update_var(inf):
    ###print'######################################################')
    ##########print (dash.callback_context.triggered[0]['prop_id'], currentframe().f_lineno)
    ###print'######################################################')
    return ''
        
@app.callback(Output('refresh__', 'children'),
              [Input('Refresh_error', 'n_clicks')])
def display_page(n):
    if n:
        shutdown()
    
    
  
@app.callback( [
                Output('status', 'children'), 
                #Output("progress_field", "children"),
                Output("toggle-add-Progress", "header"),
                # Output("Progress_select", "style"),
                Output("start_text_continue", "disabled"),
                Output("start_text_input", "disabled"),
                Output("face_type_select", "disabled"),
                #Output("head", "disabled"),
                #Output("half_face", "disabled"),
                Output("modal_error_details", "children"),
                Output("modal_error", "is_open"),
                Output("interval-1", "interval"),
                #Output("toggle-add-face", "is_open"),
                #Output("all_imgs_faces", "children"),
         Output("start_buttons","style"),
         Output("preview_divs","style"),
         Output('Progress_modal_',"is_open"),
         Output('progress_msg',"children"),
          Output('Progress_modal',"value"),
          Output('choose_face_modal',"children"),
          Output('error_modal', 'is_open')
                ],
              
    [Input('start_text_continue_', 'children'),Input('interval-1', 'n_intervals'), Input('confirm_delete', 'children'),Input('temp_delete', 'children'), Input('Resetal-addclick', 'n_clicks'),
      Input('delete-addclick', 'n_clicks'), Input('convert_start', 'n_clicks')],
    [State("toggle-add-face", "is_open"), State('start_text_input', 'value'), State("start_text_input", "disabled"), State("face_type_select", "value"), State("interval-1", "interval")])
def update_start(n, intval,confirm_delete, aadss, fkdk,lsls, dddw,t1, model_name, d3, s1, s4):
  
#html.Div([html.Div(id = 'progress_msg'),dbc.Progress(value=0, id="Progress_modal", striped=True, animated = True), html.Div(id = 'choose_face_modal')])
  global threadon 
  global msglist
  global storemsg
  global src_vids_clip
  global tar_vids_clip
  global gui_queue
  global cvt_id
  ##global thread_list
  global threadon_
  global no_loop
  ##print (thread_list)
  ##print (len(thread_list))
  #is_modal_open = dash.no_update
  #Progress_modal = dash.no_update
  
  #progress_msg = dash.no_update
  #status_children = dash.no_update
  #Progress_header = dash.no_update
  start_text_continue_disabled = dash.no_update
  start_text_input_disabled = dash.no_update
  face_type_select_disabled = dash.no_update
  modal_error_details = ''
  modal_error_is_open = False
  interval_interval = dash.no_update
  #open_choose_box = open_choose_box
  error_modal = dash.no_update
  
  global open_choose_box
  trigger_id = dash.callback_context.triggered[0]['prop_id']
  #if trigger_id == 'Resetal-addclick.n_clicks': print ('abc2')
  ##print (trigger_id)
  
  if os.path.isfile('/tmp/ResourceExhaustedError'):
      error_modal = True
      
  if trigger_id=='convert_start.n_clicks':
    
      os.remove('/tmp/running')
  
  if n is not None:
  
      global watch
      #global labelsdict
      global run
      global labelsdict
      global total_src_frames
      global total_src_frames_paths
      global src_face_list
      global total_dst_frames
      global total_dst_frames_paths
      global dst_face_list
     #print (thread_list)
       
      if os.path.isfile('/tmp/start'):# and trigger_id == 'start_text_continue_.children':

        os.remove('/tmp/start')
        open('/tmp/running','w+').close()
        #if os.path.isfile('/tmp/model.txt'): os.remove('/tmp/model.txt')
        thr = Process(target = Main, args=(gui_queue, model_name,))
         
        thr.start()
        #global thread_list
        thread_list.append(thr)
        
       #print (thr.is_alive())  
            
        thr3 = Process(target = get_preview, args=(thr,))
        thr3.daemon = True
        thr3.start()
       #global thread_list
        thread_list.append(thr3)
        
        if IN_COLAB_DRIVE:
        
            thr2 = Process(target = save_workspace_data, args=())
            thr2.daemon = True
            thr2.start()
           #global thread_list
            thread_list.append(thr2)
        
        
        #threading.Thread(target=Main, args=(gui_queue,), daemon=True).start()
        
                
        watch.start()
        ###########print ( 'ddabjhjkasfawbwfbjbkwfbkfabkfbkfafbkkbaf')
        threadon = False
        
        start_text_continue_disabled = True
        start_text_input_disabled = True
        face_type_select_disabled =True
        
        #html.Div(dbc.Row([dbc.Col('Training ') , dbc.Col(title_project)], no_gutters = True))
        Progress_header = " Starting..."
        
      ##print (threadon_)
      #if not threadon_:
      
     #    cols = dash.no_update
        
      ##print (labelsdict['src_face_labels'],labelsdict['dst_face_labels'])
      if os.path.isfile('/tmp/cluster/labelsdict.npy'):
      
        ##print ('#########' +run.value)
        
        labelsdict = np.load('/tmp/cluster/labelsdict.npy',allow_pickle='TRUE').item()
        
        
          
        if len(labelsdict['src_face_labels']) <=1 and len(labelsdict['dst_face_labels']) <=1:
        
            shutil.rmtree('/tmp/cluster/')
            
        else:
           
            
        
            #src_imgs = []
            
            ##########print (labelsdict['src_face_labels'])
            
            
            #for cli in labelsdict['src_face_labels']:
                
            #    img = cv2.imread(labelsdict['src_face_labels'][cli][0])
                
            ##    ret, frame = cv2.imencode('.png', img)
              #   frame = base64.b64encode(frame)
#
              #   src_imgs.append('data:image/png;base64,{}'.format(frame.decode()))
              #   
            #dst_imgs = []
        
            #for cli in labelsdict['dst_face_labels']:
                
              #   img = cv2.imread(labelsdict['dst_face_labels'][cli][0])
                
              #   ret, frame = cv2.imencode('.png', img)
            #    frame = base64.b64encode(frame)
              #   dst_imgs.append('data:image/png;base64,{}'.format(frame.decode()))    
                
            ###########print ('#######################################################')
            ###########print (len(src_imgs))
            ###########print (lesrc_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', cv2.imread(labelsdict['src_face_labels'][0][0]))[-1]).decode())n(dst_imgs))
            
             
            total_src_frames = 0
            total_src_frames_paths = []
            src_face_list = []
            total_dst_frames = 0
            total_dst_frames_paths = []
            dst_face_list = []
            
            src_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', imutils.resize(cv2.imread(labelsdict['src_face_labels'][0][0]), height = 64))[-1]).decode())
            dst_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', imutils.resize(cv2.imread(labelsdict['dst_face_labels'][0][0]), height = 64))[-1]).decode())
            
            
            
            
            
            src_child_  = loading(dbc.Card(
                                [
                                    dbc.CardHeader(dbc.InputGroup([dbc.Select(id = 'select_src_face',
            options = [{"label":"face_"+str(idx),"value":str(idx)} for idx in range(len(labelsdict['src_face_labels']))], value = '0'),
            dbc.Button(outline=True, id = 'add_src_face', active=False, disabled = False, color="primary", className="fas fa-user-plus")], size="sm")),
                                    
                                    dbc.CardImg(src=src_img, bottom=True, id = 'src_face_img'),
                                    dcc.Slider(id = 'src_slider', min = 0, max = len(labelsdict['src_face_labels'][0]), step = 1, value = 0,
                                        marks = {int(len(labelsdict['src_face_labels'][0])/2):str(len(labelsdict['src_face_labels'][0])) + ' frames'}),
                                    dbc.CardFooter("0 frames added", id = 'src_frames_nos', ),
                                    
                                ]
                            ))
                                          
            dst_child_  = loading(dbc.Card(
                                [
                                    dbc.CardHeader(dbc.InputGroup([dbc.Select(id = 'select_dst_face',
            options = [{"label":"face_"+str(idx),"value":str(idx)} for idx in range(len(labelsdict['dst_face_labels']))], value = '0'),
            dbc.Button(outline=True, id = 'add_dst_face', active=False, disabled = False, color="primary", className="fas fa-user-plus")], size="sm")),
                                    
                                    dbc.CardImg(src=dst_img, bottom=True, id = 'dst_face_img'),
                                    dcc.Slider(id = 'dst_slider', min = 0, max = len(labelsdict['dst_face_labels'][0]), step = 1, value = 0,
                                    marks = {int(len(labelsdict['dst_face_labels'][0])/2):str(len(labelsdict['dst_face_labels'][0])) + ' frames'}),
                                    dbc.CardFooter("0 frames added", id = 'dst_frames_nos',),
                                    
                                ]
                            ))
            
            
            
            cols = html.Div([dbc.Row([dbc.Col(src_child_), dbc.Col(dst_child_), ]), html.Br(), dbc.Button('Next ', id = 'okay_face_select', active=False, color="light",  size = 'sm',  style = {'margin-left': 'auto', 'margin-right': 'auto'}),
                            dbc.Tooltip('Add to Source profile', target="add_src_face"),dbc.Tooltip('Add to Target profile', target="add_dst_face"),], id = "cols_", style = {'text-align':'center'})
            os.remove('/tmp/cluster/labelsdict.npy')
            open_choose_box = True
            threadon_ = False
            
      
      
        
        
        
      
      try:
          message = gui_queue.get_nowait()
      except:            
          message = None 
          
      #status_children  = dash.no_update  
      
      if message:
        
        ####print'fafas')
        
        #print ('#############################################')
        #print (message)
        ##print (message)
        
        if message.startswith('#ID-'):
        
            cvt_id = message
        else:
            
            msglist = message
            
      
        
     
      #print (os.path.isfile('/tmp/processing'))
      
            
      
      
      
      
      
      
          
          
      
            
            
            #return [status_children, Progress_header, start_text_continue_disabled, start_text_input_disabled, face_type_select_disabled, modal_error_details, modal_error_is_open, interval_interval, open_choose_box, cols]
         
 
      #if message:
        
       # if message.startswith(':Stopped:'):
            
        #    start_text_continue_disabled = False
        #    start_text_input_disabled = False
        #    face_type_select_disabled =False
        #    threadon = True
        #    threadon_ = True
        #    no_loop = False
        #    Progress_header = 'Choose an option'
        #    status_children = 'Start the Process'
            
      
      
      
      
      
      #return [  status_children,Progress_header, start_text_continue_disabled, start_text_input_disabled, face_type_select_disabled,  modal_error_details, modal_error_is_open, interval_interval, open_choose_box, cols]
      
  else:
  
      #status_children = 'Start the Process'
      
      #Progress_header = 'Choose an option'
      
      
      start_text_continue_disabled = False
      start_text_input_disabled = False
      face_type_select_disabled = False
      status_children = 'Start the Process'
      
      
      
      #return [  status_children, Progress_header , start_text_continue_disabled, start_text_input_disabled, face_type_select_disabled,  modal_error_details, modal_error_is_open, interval_interval, open_choose_box, cols]
    
    
  
    #Progress_header = 'Choose an option'
    #status_children = 'Start the Process'
    
  #is_modal_open = False
  
  
  
  if os.path.isfile('/tmp/processing'):
     f = open('/tmp/processing','r')
     msglist = f.read()
     f.close()
     progress_msg = msglist
     is_modal_open = True
     
     if msglist != 'Initializing':
   
         bb = msglist.split('[')[-1].split(']')[0].split('/')
         Progress_modal = int((int(bb[0])/int(bb[1]))*100)
       
     else:
         Progress_modal = 0
   
  else:    
  
     
   
     if not os.path.isfile('/tmp/running') and not os.path.isfile(datadir()+'/model/new_SAEHD_preview_SAEHD.jpg'):
     
         progress_msg = ""
         Progress_modal = 0
         
         is_modal_open = False
         
     elif os.path.isfile('/tmp/running') and not os.path.isfile(datadir()+'/model/new_SAEHD_preview_SAEHD.jpg'):
     
         progress_msg = "Training will start shortly"
         Progress_modal = 100
         
         is_modal_open = True    
     else:
         progress_msg = "Completed"
         is_modal_open = False
         Progress_modal = 100
               
    
        
  if not os.path.isfile('/tmp/running'): 
      display_ = {'text-align' : 'center'}
      display_1 = {"display":"none"}
  
  else:
      display_ =  {"display":"none"}#dash.no_update
      display_1 = {'text-align' : 'center'}
      
  if is_modal_open:
      display_ =  {"display":"none"}#dash.no_update
      display_1 = {'text-align' : 'none'}
      
  try:
       
        tessst = cols
        #del cols
  except:
  
        cols = dash.no_update
        
        
        
  if os.path.isfile('/tmp/running'):
  
  
  
    try:
  
        f = open('/tmp/model.txt','r')
        convert_id = f.read()
        f.close()
    
    except:
        convert_id = ''
    title_project = html.Div([dbc.Badge([ dbc.Spinner(size="lg", color = 'danger'), ' Training: ', dbc.Badge(convert_id,color = 'primary', id = 'status_msg')], color="light", className="ml-1")])
                    
  
    status_children = title_project
    try:
    
      header = watch.get_interval()
    
    
      try:
          iters = str(open(datadir()+'/model/iteration.txt','r').read())
        
          itt = '[Iterations : '+iters + '] '
        
      except:
          itt = ''
         
        
    
      
    except:
  
    
      header = ''
    
      itt = ''
    
   
  #Progress_header = '['+header+'] '+itt+msglist
  
  #if msglist != ':Stopped:': 
    
    Progress_header = '[Time : '+header+'] '+itt
    
  #else:
  
   # Progress_header = ' Stopped'
  else:
      Progress_header = 'Choose an option'
      status_children = 'Start the Process'
        
        
  if trigger_id == 'Resetal-addclick.n_clicks' or trigger_id == 'delete-addclick.n_clicks' or trigger_id == 'convert_start.n_clicks':
    
    
    start_text_continue_disabled = False
    start_text_input_disabled = False
    face_type_select_disabled =False
    threadon = True
    threadon_ = True
    no_loop = False
    display_ = {'text-align' : 'center'}
    display_1 = {"display":"none"}
        
          
  ##print(interval_interval)    
  return [  status_children, Progress_header , start_text_continue_disabled, start_text_input_disabled, face_type_select_disabled,  modal_error_details, modal_error_is_open, interval_interval, display_, display_1, is_modal_open,progress_msg, Progress_modal,cols,error_modal]
    
@app.callback([Output('src_face_img', 'src'),Output('src_frames_nos', 'children'), Output('add_src_face', 'disabled'), Output('src_slider', 'max'), Output('src_slider', 'marks')],
            [Input('select_src_face', 'value'), Input('add_src_face', 'n_clicks'), Input('src_slider', 'value')])
            
def update(faceid, n, k):
    
    global labelsdict
    global total_src_frames
    global total_src_frames_paths
    global src_face_list 
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    
    
    if trigger_id == 'select_src_face.value':
    
        k = 0 
    
    ##########print (k)
    
    try:
        src_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', imutils.resize(cv2.imread(labelsdict['src_face_labels'][int(faceid)][k]), height = 64))[-1]).decode())
    except:
        src_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', imutils.resize(cv2.imread(labelsdict['src_face_labels'][int(faceid)][0]), height = 64))[-1]).decode())
        
            
            
    n_frames = len(labelsdict['src_face_labels'][int(faceid)])
    ##########print (n_frames)
    
    if n and trigger_id == 'add_src_face.n_clicks':
        total_src_frames = total_src_frames + n_frames
        for i in labelsdict['src_face_labels'][int(faceid)]:
            total_src_frames_paths.append(i)
        
        src_face_list.append(faceid)
    
    if faceid in src_face_list:
        isdisabled = True
    else:
        isdisabled = False
    ##########print (src_face_list, faceid, isdisabled)
    return src_img, str(total_src_frames) + ' frames added', isdisabled, n_frames, {int(n_frames/2):str(n_frames) + ' frames'}
        
    
    
@app.callback([Output('dst_face_img', 'src'),Output('dst_frames_nos', 'children'), Output('add_dst_face', 'disabled'),Output('dst_slider', 'max'), Output('dst_slider', 'marks')],
            [Input('select_dst_face', 'value'), Input('add_dst_face', 'n_clicks'), Input('dst_slider', 'value')])
            
def update(faceid, n, k):
    
    global labelsdict
    global total_dst_frames
    global total_dst_frames_paths
    global dst_face_list 
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    
    if trigger_id == 'select_dst_face.value':
        ###print'ss')
    
        k = 0
        
    ##########print (k)
    try:
        dst_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', imutils.resize(cv2.imread(labelsdict['dst_face_labels'][int(faceid)][k]), height = 64))[-1]).decode())
    except:
    
        dst_img = 'data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png', imutils.resize(cv2.imread(labelsdict['dst_face_labels'][int(faceid)][0]), height = 64))[-1]).decode())
    
    n_frames = len(labelsdict['dst_face_labels'][int(faceid)])
    ##########print (n_frames)
    
    if n and trigger_id == 'add_dst_face.n_clicks':
        total_dst_frames = total_dst_frames + n_frames
        for i in labelsdict['dst_face_labels'][int(faceid)]:
            total_dst_frames_paths.append(i)
        
        dst_face_list.append(faceid)
    
    if faceid in dst_face_list:
        isdisabled = True
    else:
        isdisabled = False
    ##########print (dst_face_list, faceid, isdisabled)
    return dst_img, str(total_dst_frames) + ' frames added', isdisabled, n_frames, {int(n_frames/2):str(n_frames) + ' frames'}
                
@app.callback([Output('confirm_delete', 'children'),Output('okay_face_select_text', 'children'), 
                Output('okay_face_select_text', 'disabled'), Output('select_src_face', 'disabled'), Output('select_dst_face', 'disabled'), Output('src_slider', 'disabled'),
                Output('dst_slider', 'disabled'),Output('cols_', 'style')],
              [Input('okay_face_select', 'n_clicks')])
def update(n):
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    if n and trigger_id == 'okay_face_select.n_clicks':
        global total_dst_frames_paths
    
        global total_src_frames_paths
        
        if len(total_dst_frames_paths) >0 and len(total_src_frames_paths)>0:
        
            all_src_files = glob.glob(datadir()+'/data_src/aligned/*')
            
            all_src_files_delete = set(all_src_files) - set(total_src_frames_paths)
            ##########print (total_src_frames_paths)
            ##########print (all_src_files_delete)
            for i in all_src_files_delete:
            
                os.remove(i)
                
            all_dst_files = glob.glob(datadir()+'/data_dst/aligned/*')
            
            all_dst_files_delete = set(all_dst_files) - set(total_dst_frames_paths)
            ##########print (all_dst_files_delete)
            for i in all_dst_files_delete:
            
                os.remove(i)  
                
            shutil.rmtree('/tmp/cluster/')
            return " ", '', True, True,True,True,True,{"display":"none"}
            
        else:
        
            
            return dash.no_update, 'Please add frames', dash.no_update, dash.no_update, dash.no_update , dash.no_update, dash.no_update, dash.no_update
    
    else:
    
        
        return [dash.no_update]*8
    
@app.callback(Output('Convert_Image', 'src'),
                [
                Input('v_plus_size', 'n_clicks'),
                Input('h_minus_size', 'n_clicks'),
                Input('h_plus_size', 'n_clicks'),
                Input('v_minus_size', 'n_clicks'),
                Input('v_plus_shift', 'n_clicks'),
                Input('h_minus_shift', 'n_clicks'),
                Input('h_plus_shift', 'n_clicks'),
                Input('v_minus_shift', 'n_clicks'),
                Input('refresh_img', 'n_clicks'),
            
                Input('mask_mode_', 'value'),
                #Input('face_type_', 'value'),
                Input('mode_', 'value'),
                
                Input('Erode_', 'value'),
                Input('Blur_', 'value'),
                Input('color_mode_', 'value'),
                Input('motion_blur_power_', 'value'),
                Input('blursharpen_amount_', 'value'),
                Input('image_denoise_power_', 'value'),
                Input('color_degrade_power_', 'value'),
                Input('okay_merge', 'n_clicks'), 
                Input('convert_tabs_1', 'active_tab'),
                Input('default_pre', 'n_clicks'),
                Input('ori_pre', 'n_clicks'),
                Input('default_face_pre', 'n_clicks'),
                Input('ori_face_pre', 'n_clicks'),
        ],# [State('size_step', 'value'),
          #   State('shift_step', 'value')]
                )    
                
def update_convert_image(v_plus_size,h_minus_size, h_plus_size, v_minus_size , v_plus_shift, h_minus_shift, h_plus_shift, v_minus_shift, refresh_img,
                          mask_mode_, mode_, Erode_,Blur_ ,color_mode_, motion_blur_power_, blursharpen_amount_,
                        image_denoise_power_,color_degrade_power_,okay_merge, jdkd, default_pre, ori_pre,default_face_pre,  ori_face_pre):
    
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    stp_size, stp_shift = 10,10
    ##########print (v_plus_size,h_minus_size, h_plus_size, v_minus_size , v_plus_shift, h_minus_shift, h_plus_shift, v_minus_shift,
  
    global horizontal_shear
    global vertical_shear
    global horizontal_shift
    global vertical_shift
    global ind_preview
    global npy_files
    global cfg_merge
    global show_mode
    
    
    
    if trigger_id == 'v_plus_size.n_clicks':
    
        vertical_shear = vertical_shear + stp_size
        
    if trigger_id == 'h_minus_size.n_clicks':
    
        horizontal_shear = horizontal_shear - stp_size
        
        
    if trigger_id == 'v_minus_size.n_clicks':
    
        vertical_shear = vertical_shear - stp_size
        
    if trigger_id == 'h_plus_size.n_clicks':
    
        horizontal_shear = horizontal_shear + stp_size
        
        
    if trigger_id == 'v_plus_shift.n_clicks':
    
        vertical_shift = vertical_shift + stp_shift
        
    if trigger_id == 'h_minus_shift.n_clicks':
    
        horizontal_shift = horizontal_shift - stp_shift
        
        
    if trigger_id == 'v_minus_shift.n_clicks':
    
        vertical_shift = vertical_shift - stp_shift
        
    if trigger_id == 'h_plus_shift.n_clicks':
    
        horizontal_shift = horizontal_shift + stp_shift
        
        
    npy_files = [i for i in os.listdir('/tmp') if i.endswith('.npy')]
    
    
    if trigger_id == 'refresh_img.n_clicks':
    
        ind_preview = np.random.choice(len(npy_files))
    
    
    
    try:
        npy_ = os.path.join('/tmp', npy_files[ind_preview])
    
        
        ###########print (npy_)
        
        
        if trigger_id == 'default_pre.n_clicks':
            show_mode = 1
        
        if trigger_id == 'ori_pre.n_clicks':
            show_mode = 2
        
        if trigger_id == 'default_face_pre.n_clicks':
            
            show_mode = 3
        if trigger_id == 'ori_face_pre.n_clicks':
            show_mode = 4
        
        
      
            
        
        cfg_merge = merging_vars(
               # face_type = face_type,
                mask_mode = int(mask_mode_),
                mode = mode_,
                erode_mask_modifier = Erode_,
                blur_mask_modifier = Blur_,
                color_transfer_mode = int(color_mode_),
                masked_hist_match = True,
                hist_match_threshold = 255,
                motion_blur_power = motion_blur_power_,
                blursharpen_amount = blursharpen_amount_,
                image_denoise_power = image_denoise_power_,
                bicubic_degrade_power = 0,
                sharpen_mode = 1,
                color_degrade_power = color_degrade_power_,
                horizontal_shear = horizontal_shear,
                vertical_shear = vertical_shear,
                horizontal_shift = horizontal_shift,
                vertical_shift = vertical_shift,
                show_mode = show_mode
                )
        
        
        
        
        from merger import Merger_tune
                
        result = Merger_tune.MergeMaskedFace_test(npy_, cfg_merge)
        
        ##print (result.shape)
        
        if trigger_id == 'okay_merge.n_clicks':
        
            dict_1 = {'original':0, 'overlay':1, 'hist-match':2 ,'seamless':3 ,'seamless-hist-match':4 , 'raw-rgb':5 , 'raw-predict':6}
            
            dict_2 = {0: 'None', 1:'rct',2:'lct',3:'mkl',4:'mkl-m',5:'idt',6:'idt-m',7:'sot-m',8:'mix-m'}
            
            with open('DeepFaceLab/settings.py', 'a') as f:
                f.write("\nmerging_mode = "+ str(dict_1[cfg_merge.mode]))
                f.write("\nmask_merging_mode = " + str(cfg_merge.mask_mode))
                f.write("\nblursharpen_amount = " + str(cfg_merge.blursharpen_amount))
                f.write("\nerode_mask_modifier = "+ str(cfg_merge.erode_mask_modifier))
                f.write("\nblur_mask_modifier ="+ str(cfg_merge.blur_mask_modifier))
                f.write("\nmotion_blur_power = "+ str(cfg_merge.motion_blur_power))
                #f.write("\noutput_face_scale = "+ cfg_merge)
                if cfg_merge.color_transfer_mode == 0:
                  f.write("\ncolor_transfer_mode = None")
                else:
                  f.write("\ncolor_transfer_mode = '"+ dict_2[cfg_merge.color_transfer_mode]+"'")
                #f.write("\nsuper_resolution_power = "+ cfg_merge)
                f.write("\nimage_denoise_power = "+ str(cfg_merge.image_denoise_power))
                #f.write("\nbicubic_degrade_power = "+ cfg_merge)
                f.write("\ncolor_degrade_power = "+ str(cfg_merge.color_degrade_power))
                #f.write("\nmasked_hist_match = "+ cfg_merge)
                #f.write("\nhist_match_threshold ="+cfg_merge)
                f.write("\nhorizontal_shear = "+str(cfg_merge.horizontal_shear))
                f.write("\nvertical_shear = "+ str(cfg_merge.vertical_shear))
                f.write("\nhorizontal_shift = "+ str(cfg_merge.horizontal_shift))
                f.write("\nvertical_shift = "+ str(cfg_merge.vertical_shift))
                
                f.close()
            
        
        result = imutils.resize(result*255, height=512)
        
        ##########print (result.shape)
        
        ret, frame = cv2.imencode('.png',result )
        frame = base64.b64encode(frame)
        src = 'data:image/png;base64,{}'.format(frame.decode())
        
        return src
    
    except:
        return ""
  
    
        
@app.callback([Output('preview_progress', 'value'),Output('video-player', 'url'), Output('toggle-add-right_frame', 'header')],
            [Input('interval-1', 'n_intervals')])
            
      
def update__(interval):
    try:
    
        ##print ('f')
        number_of_files = len(os.listdir(datadir()+'/preview/merged'))
        total_number_of_files = len(os.listdir(datadir()+'/preview/')) - 3
        
        done =  int((number_of_files/total_number_of_files)*100)
        ##print (done)
        
        
        try:
            secss = time.time() - os.path.getctime(glob.glob("assets/*mp4")[0])
            sec_s = ' [Updated ' +str(int(secss)//60)  + ' minutes ago]'
        
            
        except:
            sec_s = ''
        
        try:
        
            ##print (time.time() - os.path.getctime(glob.glob("assets/*mp4")[0]))
            if secss <=20:
            
                ##print ('updates')
                
                return [done, os.path.join('assets', glob.glob("assets/*mp4")[0].split('/')[-1]),sec_s]
            
            else:
                return [done, dash.no_update,sec_s]
                
        except:
        
            return [done, dash.no_update, sec_s]
            
            
    except:
    
        return [dash.no_update,dash.no_update,dash.no_update]
            
@app.callback([Output('merge_progress', 'value'),Output('convert_result', 'children'), Output('merge_progress_modal', 'is_open'),Output('merge_progress_exit', 'style'),Output('ModalHeader_convert', 'children') ],
        [Input('convert_start', 'n_clicks'), Input('merge_progress_exit', 'n_clicks'), Input('interval-1', 'n_intervals')])
        
  
def update__(nd,ne, interval):
    trigger_id = dash.callback_context.triggered[0]['prop_id']
    ModalHeader_convert = dash.no_update
    merge_progress_modal = dash.no_update
    global cfg_merge 
    #done = 0
    #global convert_id
    if nd:
        
        f = open('/tmp/model.txt','r')
        convert_id = f.read()
        f.close()
        tar_di = os.path.join(datadir(), convert_id + '.mp4')
        ##print (nd)
    
    if ne and trigger_id=='merge_progress_exit.n_clicks':
        
        os.remove('/tmp/converting')
        merge_progress_modal = False
        return dash.no_update, dash.no_update, merge_progress_modal,dash.no_update, ModalHeader_convert
         
    
    if nd and trigger_id=='convert_start.n_clicks':
    
        open('/tmp/converting','w+').close()
        ModalHeader_convert = 'In Progress'
        #ModalHeader_convert = html.Div([dbc.Badge([ dbc.Spinner(size="lg", color = 'danger'), ' Swaping: ', dbc.Badge(convert_id,color = 'primary', id = 'status_msg')], color="light", className="ml-1")])
        killall()
        ##print ('gb')
        
        #open('/tmp/converting','w+').close()
        
        for i in thread_list:
            i.terminate()
    
        if os.path.isdir(datadir()+'/data_dst/merged'):
            shutil.rmtree(datadir()+'/data_dst/merged')
            os.mkdir (datadir()+'/data_dst/merged')
            
            
        dict_1 = {'original':0, 'overlay':1, 'hist-match':2 ,'seamless':3 ,'seamless-hist-match':4 , 'raw-rgb':5 , 'raw-predict':6}
        
        dict_2 = {0: 'None', 1:'rct',2:'lct',3:'mkl',4:'mkl-m',5:'idt',6:'idt-m',7:'sot-m',8:'mix-m'}
        
        with open('DeepFaceLab/settings.py', 'a') as f:
            f.write("\nmerging_mode = "+ str(dict_1[cfg_merge.mode]))
            f.write("\nmask_merging_mode = " + str(cfg_merge.mask_mode))
            f.write("\nblursharpen_amount = " + str(cfg_merge.blursharpen_amount))
            f.write("\nerode_mask_modifier = "+ str(cfg_merge.erode_mask_modifier))
            f.write("\nblur_mask_modifier ="+ str(cfg_merge.blur_mask_modifier))
            f.write("\nmotion_blur_power = "+ str(cfg_merge.motion_blur_power))
            #f.write("\noutput_face_scale = "+ cfg_merge)
            if cfg_merge.color_transfer_mode == 0:
              f.write("\ncolor_transfer_mode = None")
            else:
              f.write("\ncolor_transfer_mode = '"+ dict_2[cfg_merge.color_transfer_mode]+"'")
            #f.write("\nsuper_resolution_power = "+ cfg_merge)
            f.write("\nimage_denoise_power = "+ str(cfg_merge.image_denoise_power))
            #f.write("\nbicubic_degrade_power = "+ cfg_merge)
            f.write("\ncolor_degrade_power = "+ str(cfg_merge.color_degrade_power))
            #f.write("\nmasked_hist_match = "+ cfg_merge)
            #f.write("\nhist_match_threshold ="+cfg_merge)
            f.write("\nhorizontal_shear = "+str(cfg_merge.horizontal_shear))
            f.write("\nvertical_shear = "+ str(cfg_merge.vertical_shear))
            f.write("\nhorizontal_shift = "+ str(cfg_merge.horizontal_shift))
            f.write("\nvertical_shift = "+ str(cfg_merge.vertical_shift))
            
            f.close()
                
                
                
                
        if os.path.isfile(tar_di): os.remove(tar_di)
        
        thr = Process(target = Convert, args=())
        #thr.daemon=True   
        thr.start()
        merge_progress_modal = True
        
        return 0, ["Loading video frames", ]   , merge_progress_modal, {'display':'none'}, ModalHeader_convert
    
    if  os.path.isfile('/tmp/converting'):
    
        try:
        
        
            ##print (trigger_id, nd); 
            number_of_files = len(os.listdir(datadir()+'/data_dst/merged'))
            total_number_of_files = len(os.listdir(datadir()+'/data_dst/'))-2 
            
            done =  int((number_of_files/total_number_of_files)*100)
            
            
        
            if os.path.isfile(tar_di):
                #time.sleep(10)
            
                #fid = getoutput("xattr -p 'user.drive.id' '"+tar_di+"'")
                
                done = 100
                
                done_ = [html.Br(),"Goto path "+tar_di+' to play it']
                
                sty = {'display':''}
                ModalHeader_convert = 'Completed'
       
            
            else:
                done_ =  [ "Please wait until we finish the swaping process"]   
                sty =  {'display':'none'}
                
            
            merge_progress_modal = os.path.isfile('/tmp/converting')
            
            
            return done,done_, merge_progress_modal, sty, ModalHeader_convert
            
        except:
            pass
            
    merge_progress_modal = os.path.isfile('/tmp/converting')
    
    return 0, "", merge_progress_modal, {'display':'none'}, ModalHeader_convert
    
    
if __name__ == '__main__':
    
    
    app.run_server(debug=False, port =  4000, host = '0.0.0.0')
#gunicorn app:server -b 0.0.0.0:8080