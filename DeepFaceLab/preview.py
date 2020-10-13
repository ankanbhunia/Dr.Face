import glob 
import numpy as np
from shutil import copyfile
import os
def datadir():
     f = open('/tmp/model.txt','r')
     n = f.read()
     return os.path.join('/data', n)
    

if os.path.isdir(datadir()+'/preview'): os.system('rm -r '+datadir()+'/preview')

import os

if not os.path.isdir(datadir()+'/preview'): os.mkdir(datadir()+'/preview')
if not os.path.isdir(datadir()+'/preview/aligned'): os.mkdir(datadir()+'/preview/aligned')
if not os.path.isdir(datadir()+'/preview/merged'): os.mkdir(datadir()+'/preview/merged')
if not os.path.isdir(datadir()+'/preview/merged_mask'): os.mkdir(datadir()+'/preview/merged_mask')

f = glob.glob(datadir()+'/data_dst/aligned/*')
if len(f)>50:
  h = np.arange(0,50,2)
  f = np.array(sorted(f)[:50])[h]

else:
  h = np.arange(0,len(f),2)
  f = np.array(sorted(f)[:len(f)])[h]

for i in f:
  copyfile(i, os.path.join(datadir()+'/preview/aligned/', i.split('/')[-1]))
  copyfile(os.path.join(datadir()+'/data_dst/',i.split('/')[-1].split('_')[0]+'.png'), 
           os.path.join(datadir()+'/preview/', i.split('/')[-1].split('_')[0]+'.png'))

