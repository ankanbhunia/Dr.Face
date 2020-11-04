import os
from IPython.display import HTML, clear_output
from IPython.display import Javascript
from IPython.display import Image
from google.colab.output import eval_js
import argparse

parser = argparse.ArgumentParser(description='Dr.Face Options')


parser.add_argument('--ngrok',action='store_true',
                   help='Use ngrok as tunnel medium')

parser.add_argument('--token', type=str,
                   help='Enter ngrok Authtoken from https://dashboard.ngrok.com/auth/your-authtoken ')
parser.add_argument('--path', type=str,
                    help='Specify drive path')
                    
                    
parser.add_argument('--no_drive', action='store_true',
                    help='don\'t mount drive')  

parser.add_argument('--debug', action='store_true',
                    help='the shell output will be shown')

args = parser.parse_args()

xxx = args.ngrok

token_ = args.token
drive_path = args.path
notmount = args.no_drive
debug = args.debug

    
import sys
clear_output()

ipy = get_ipython()
ipy.magic("tensorflow_version 1.x")

clear_output()

  
GPU = get_ipython().getoutput("nvidia-smi --query-gpu=name --format=csv,noheader")

try:
  gpu = GPU[0]
except:
  gpu = 'CPU'



if not os.path.isfile('/tmp/done'):
  if not notmount:
    if not os.path.isdir('drive/'):
      from google.colab import drive; drive.mount('drive', force_remount=True) 
      
      clear_output()
      
     

  print ('['+gpu+']'+' Please wait for few minutes... ')
  
  print ('\n[1/3] Downloading repository from Github')
  get_ipython().system_raw('git clone https://github.com/ankanbhunia/Dr.Face.git foo; mv foo/* foo/.git* .; rmdir foo')
  print ('\n[2/3] Downloading python environment')
  O_ = get_ipython().getoutput('gdown --id 1-7XxOStSUCRfjtgJqzO57Z2Ku2TpmBGa')
  print ('\n[3/3] Setting up environment for Dr.Face')
  get_ipython().system_raw('tar -xvf drfacelib.tar.gz; rm drfacelib.tar.gz; touch /tmp/done')
  
  #get_ipython().system_raw('git clone https://github.com/ankanbhunia/Dr.Face.git foo; mv foo/* foo/.git* .; rmdir foo; gdown --id 1-7XxOStSUCRfjtgJqzO57Z2Ku2TpmBGa; tar -xvf drfacelib.tar.gz; rm drfacelib.tar.gz; touch /tmp/done')
  get_ipython().system_raw('sudo apt-get install -y xattr')

clear_output()




if xxx:

  try:

    get_ipython().system_raw("pip3 install pyngrok")
    #get_ipython().system_raw("ngrok authtoken " + xxx)
    from pyngrok import ngrok
    if token_:
      ngrok.set_auth_token(token_)
    ngrok.kill()
    print("[GPU Device]="+gpu, end= ' | '+"Project URL: "+ngrok.connect(4000).public_url)

  except:

    print ('The ngrok token is invalid')
    print("[GPU Device]="+gpu, end= ' | '+"Project URL: "+eval_js("google.colab.kernel.proxyPort(%d)"% (4000)))

else:

  print("[GPU Device]="+gpu, end= ' | '+"Project URL: "+eval_js("google.colab.kernel.proxyPort(%d)"% (4000)))



get_ipython().system_raw("fuser -k 4000/tcp")

if drive_path:

    if not debug:
      get_ipython().system_raw("Library/bin/python app.py "+drive_path)
    else:
          
      print("""
                                                                                                                                                                    
      """)
      G = get_ipython().getoutput("Library/bin/python app.py "+drive_path)
      
else:

    if not debug:
      get_ipython().system_raw("Library/bin/python app.py")
      
    else:

          
      print("""
                                                                                                                                                                    
      """)
      G = get_ipython().getoutput("Library/bin/python app.py")
