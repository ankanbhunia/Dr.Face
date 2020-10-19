

def Get_face_clustered_labels(faces_folder_path):

	


    import sys
    import os
    import dlib
    import glob
    from tqdm import tqdm
	
    #print (faces_folder_path, os.listdir(faces_folder_path))

    # Download the pre trained models, unzip them and save them in the save folder as this file
    predictor_path = 'FaceClust/shape_predictor_5_face_landmarks.dat' # Download from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    face_rec_model_path = 'FaceClust/dlib_face_recognition_resnet_model_v1.dat' # Download from http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2


    detector = dlib.cnn_face_detection_model_v1('FaceClust/mmod_human_face_detector.dat') #a detector to find the faces
    sp = dlib.shape_predictor(predictor_path) #shape predictor to find face landmarks
    facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model

    descriptors = []
    images = []

    # Load the images from input folder

    FACE_PATHS  = []
    dn = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    f = open('/tmp/ProgressN','w+')
    f.write(str(len(dn)))
    f.close()
    ccc = 1
    for f in tqdm(dn):
        ff = open('/tmp/ProgressI','w+')
        ff.write(str(ccc))
        ff.close()
        ccc = ccc + 1
        #print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        # Ask the detector to find the bounding boxes of each face. The 1 in the second argument indicates that we should upsample the image 1 time. This will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        
        #print("Number of faces detected: {}".format(len(dets)))
         
        # Now process each face we found.
        
        for k, d in enumerate(dets):
        
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d.rect)
            
            # Compute the 128D vector that describes the face in img identified by shape.  
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            
            
            
            descriptors.append(face_descriptor)
            
            images.append((img, shape))
            
            FACE_PATHS.append(f)

    # Cluster the faces.  
    labels = dlib.chinese_whispers_clustering(descriptors, 0.5)

    num_classes = len(set(labels)) # Total number of clusters

    print("Number of clusters: {}".format(num_classes))

    face_label_dicts = {}
    
    #print ('xxxxxx')
    #print (FACE_PATHS)
    for i,j in zip(FACE_PATHS, labels):

        if j in face_label_dicts:
        
            face_label_dicts[j].append(i)
            
        else:
            face_label_dicts[j] = [i]
            
    #print ('xxxxxx')	
    
    if os.path.isfile('/tmp/ProgressN'): os.remove('/tmp/ProgressN')
    if os.path.isfile('/tmp/ProgressI'): os.remove('/tmp/ProgressI')
    del detector
    del sp
    del facerec
    del dlib
    return face_label_dicts
    



