"""import streamlit as st
#from views.objectDetection import *
import os  
import numpy as np

from pathlib import Path
import configparser
import cv2
import numpy as np
import threading

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#from views import video_utils


import pickle
import sys
"""

""" from views.app import *
#---------------------------------------
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image 
import PIL 
import tqdm

import time  """
#_____________________

import streamlit as st
#weight = "C:/Users/DELL Latitude 7270/Desktop/Les meilleurs modeles/detectron2 coco 40000it et 21%/output_detectron22_coco-20220905T201956Z-001/output_detectron22_coco/model_final.pth"
#kitti_metadata = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/Les meilleurs modeles/detectron2 coco 40000it et 21%/output_detectron22_coco-20220905T201956Z-001/output_detectron22_coco/kitti.dat", "rb"))
#cfg = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/Les meilleurs modeles/detectron2 coco 40000it et 21%/output_detectron22_coco-20220905T201956Z-001/output_detectron22_coco/cfg.dat", "rb"))



#weight = "C:/Users/DELL Latitude 7270/Desktop/LAST MODELS/coco D2 10.000 40%/output_detectron22_coco-20220910T080244Z-001/output_detectron22_coco/model_final.pth"
#kitti_metadata = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/LAST MODELS/coco D2 10.000 40%/output_detectron22_coco-20220910T080244Z-001/output_detectron22_coco/kitti.dat", "rb"))
#cfg = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/LAST MODELS/coco D2 10.000 40%/output_detectron22_coco-20220910T080244Z-001/output_detectron22_coco/cfg.dat", "rb"))

#weight = "views/model_final.pth"
#kitti_metadata = pickle.load(open("views/kitti.dat", "rb"))
#cfg = pickle.load(open("views/cfg.dat", "rb"))

#weight = "C:/Users/DELL Latitude 7270/Desktop/LAST MODELS/3000 it kitti detecton2 39 et qlq dernier qui est dans drive de kitti d2/output_detectron2-20220910T075025Z-001/output_detectron2/model_final.pth"
#kitti_metadata = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/LAST MODELS/3000 it kitti detecton2 39 et qlq dernier qui est dans drive de kitti d2/output_detectron2-20220910T075025Z-001/output_detectron2/kitti.dat", "rb"))
#cfg = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/LAST MODELS/3000 it kitti detecton2 39 et qlq dernier qui est dans drive de kitti d2/output_detectron2-20220910T075025Z-001/output_detectron2/cfg.dat", "rb"))
#claas detector
"""class Detector:



    def __init__(self, model_type = "objectDetection"):
        self.cfg = cfg
        #self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        #self.cfg.MODEL.WEIGHTS = "C:/Users/DELL Latitude 7270/Desktop/new out/output_detectron2-20220827T183103Z-001/output_detectron2/model_final.pth"
        self.cfg.MODEL.WEIGHTS = weight
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cpu" # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)

        ##Time Time##
        
        start_time = time.time()

        predictions = self.predictor(image)

        delta = time.time() - start_time


        ##end Time Time

        viz = Visualizer(image[:,:,::-1],metadata= kitti_metadata ,scale=1.2)
        
        output = viz.draw_instance_predictions(predictions['instances'].to('cpu'))
        filename = 'result.jpg'
        cv2.imwrite(filename, output.get_image()[:,:,::-1])
        #cv2.imshow("Result",output.get_image()[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Temps d'exécution en secondes:")
        print(delta)
        st.write("Temps d'exécution en secondes:")
        st.write(delta)


    def onVideo(self,videoPath):
        start_time = time.time()

        video = cv2.VideoCapture(videoPath)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        #fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        #out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))

        #initializing videoWriter
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter("output.mp4", fourcc , fps=float(frames_per_second), frameSize=(width, height), isColor=True)

       
        v = VideoVisualizer(kitti_metadata, ColorMode.IMAGE)

        

        def runOnVideo(video,maxFrames):# this is for debugging
            
            readFrames = 0
            while True:
                hasFrame , frame = video.read()
                if not hasFrame:
                    break

                
                outputs = self.predictor(frame)
                
                 
                

                frame  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                visualization = v.draw_instance_predictions(frame, outputs['instances'].to('cpu'))
                visualization = cv2.cvtColor(visualization.get_image(),cv2.COLOR_RGB2BGR)
                yield visualization
                
                readFrames += 1
                if readFrames > maxFrames:
        
                    break
                   
                
        num_frames = 200 # here  we reinitialize the number of frames because  of it'll take hours to write the detectedVideos to our video_file and since we don't have a gpu
        # if num_frames is not re-inititialized. the entire frames of the video will be taken into account.. usually taking hours to detect since a 'cpu' and not'gpu' is used
        for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total = num_frames):
            #cv2.imwrite("detected_image.png", visualization)

            video_writer.write(visualization)
        video.release()
        video_writer.release()
        cv2.destroyAllWindows()
        delta = time.time() - start_time 
        print("Temps d'exécution en secondes:")
        print(delta)
        st.write("Temps d'exécution en secondes:")
        st.write(delta)





















#----------------
#Load the variable
#kitti_metadata = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/new out/output_detectron2-20220827T183103Z-001/output_detectron2/kitti.dat", "rb"))
#Load the variable
#cfg = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/new out/output_detectron2-20220827T183103Z-001/output_detectron2/cfg.dat", "rb"))
#Load the variable
#kitti_dataset = pickle.load(open("C:/Users/DELL Latitude 7270/Desktop/out/output_detectron2/kitti_dataset.dat", "rb"))

#detector = Detector(model_type='keypointsDetection')

#detector.onVideo("pexels-tima-miroshnichenko-6388396.mp4")
#@st.cache
def func_1(x):
    detector = Detector(model_type=x)
    image_file = st.file_uploader("Téléchargement de l'image",type=['png','jpeg','jpg'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        st.write(file_details)
        img = Image.open(image_file)
        st.image(img, caption='Image téléchargée.')
        with open(image_file.name,mode = "wb") as f: 
            f.write(image_file.getbuffer())         
        st.success("Fichier enregistré")
        detector.onImage(image_file.name)
        img_ = Image.open("result.jpg")
        st.image(img_, caption='Image traitée.')

def func_2(x):
    detector = Detector(model_type=x)
    uploaded_video = st.file_uploader("Téléchargement de la video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk
    
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Vidéo téléchargée")
        detector.onVideo(vid)
        st_video = open('output.mp4','rb')
        video_bytes = st_video.read()

        
        input_file = 'output.mp4' # You should enter the file's path here
        # Using fstring for variable file name
        # Command to execute 
        # Using Windows OS command 
        cmd = f'ffmpeg -i {input_file} -vcodec libx264 play.mp4'
        # Using os.system() method 
        os.system(cmd) 


        st_video = open('play.mp4','rb')
        video_bytes = st_video.read()


        st.video(video_bytes)
        st.write("Vidéo détectée")
#--------------------------------------------------------------------------------------------------------------------------------------
#webcam detection
#@st.cache
def video_stream(video_source):
    #video_source = available_cameras[cam_id]
    video_thread = video_utils.WebcamVideoStream(video_source)
    video_thread.start()

    return video_thread

"""
"""
@st.cache(persist=True)
def initialization():"""
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        classes_names (list: str): Classes available for the model of interest.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    #cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    #cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # Set threshold for this model
 """
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = weight
    # Get classes names for the dataset of interest
    classes_names = kitti_metadata.thing_classes
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, classes_names, predictor


def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], kitti_metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image() #[:, :, ::-1]

    return processed_img


@st.cache"""
"""
def discriminate(outputs, classes_to_detect):"""
    """Select which classes to detect from an output.

    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them

    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes", 
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.

    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.

    """
   """ print('aaaaa')
    print(outputs['instances'].pred_classes)
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, classes_to_detect)
    # Get the indexes
    idx = np.nonzero(mask)
"""
    # # Get the current Instance values
    # pred_boxes = outputs['instances'].pred_boxes
    # pred_classes = outputs['instances'].pred_classes
    # pred_masks = outputs['instances'].pred_masks
    # scores = outputs['instances'].scores

    # Get Instance values as a dict and leave only the desired ones
    """
    out_fields = outputs['instances'].get_fields()
    for field in out_fields:
        out_fields[field] = out_fields[field][idx]

    return outputs"""








def main2():
    # Initialization
   """ cfg, classes, predictor = initialization()

    # Streamlit initialization
    #st.title("Instance Segmentation")
    #st.sidebar.title("Options")
    ## Select classes to be detected by the model
    #classes_to_detect = st.multiselect(
     #   "choisir classes", classes, ['person'])
    #mask = np.isin(classes, classes_to_detect)
    #class_idxs = np.nonzero(mask)
    ## Select camera to feed the model
    available_cameras = {'Camera 1': 0}
    cam_id = st.selectbox(
        "Choisir camera", list(available_cameras.keys()))

    # Define holder for the processed image
    img_placeholder = st.empty()

    # Load video source into a thread
    
    video_source = available_cameras[cam_id]
    video_thread = video_stream(video_source)"""
    # video_thread = video_utils.WebcamVideoStream(video_source)
    # video_thread.start()
    
    # Detection code
    """
    try:
        while not video_thread.stopped():
            # Camera detection loop
            frame = video_thread.read()
            if frame is None:
                print("Frame stream interrupted")
                break
            # Change color gammut to feed the frame into the network
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detection code
            outputs = inference(predictor, frame)
            #outputs = discriminate(outputs, class_idxs)
            out_image = output_image(cfg, frame, outputs)
            #st.image(out_image, caption='Processed Image', use_column_width=True)        
            img_placeholder.image(out_image)





            # output = run_inference_for_single_image(frame, sess, 
            #     detection_graph)
            # output = discriminate_class(output, 
            #     classes_to_detect, category_index)
            # processed_image = visualize_results(frame, output, 
            #     category_index)

            # # Display the image with the detections in the Streamlit app
            # img_placeholder.image(processed_image)
            
            # #cv2.imshow('Video', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            # # if cv2.waitKey(1) & 0xFF == ord('q'):
            # #     break
    
    except KeyboardInterrupt:   
        pass

    print("Ending resources")
    st.text("Camera not detected")
    cv2.destroyAllWindows()
    video_thread.stop()
    sys.exit()


"""









    # # Initialization
    # ## Load the configuration variables from 'config.ini'
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    # ## Loading label map
    # num_classes = config.getint('net', 'num_classes')
    # path_to_labels = config['net']['path_to_labels']
    # label_map = label_map_util.load_labelmap(path_to_labels)
    # categories = label_map_util.convert_label_map_to_categories(label_map, 
    #     max_num_classes=num_classes, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    # # Streamlit initialization
    # st.title("Object Detection")
    # st.sidebar.title("Object Detection")
    # ## Select classes to be detected by the model
    # classes_names = [value['name'] for value in category_index.values()]
    # classes_names.sort()
    # classes_to_detect = st.sidebar.multiselect(
    #     "Select which classes to detect", classes_names, ['person'])
    # ## Select camera to feed the model
    # available_cameras = {'Camera 1': 0, 'Camera 2': 1, 'Camera 3': 2}
    # cam_id = st.sidebar.selectbox(
    #     "Select which camera signal to use", list(available_cameras.keys()))
    # ## Select a model to perform the inference
    # available_models = [str(i) for i in Path('./trained_model/').iterdir() 
    #     if i.is_dir() and list(Path(i).glob('*.pb'))]
    # model_name = st.sidebar.selectbox(
    #     "Select which model to use", available_models)
    # # Define holder for the processed image
    # img_placeholder = st.empty()

    # # Model load
    # path_to_ckpt = '{}/frozen_inference_graph.pb'.format(model_name)
    # detection_graph = model_load_into_memory(path_to_ckpt)

    # # Load video source into a thread
    # video_source = available_cameras[cam_id]
    # ## Start video thread
    # video_thread = video_utils.WebcamVideoStream(video_source)
    # video_thread.start()
    
    # # Detection code
    # try:
    #     with detection_graph.as_default():
    #         with tf.Session(graph=detection_graph) as sess:
    #             while not video_thread.stopped():
    #                 # Camera detection loop
    #                 frame = video_thread.read()
    #                 if frame is None:
    #                     print("Frame stream interrupted")
    #                     break
    #                 # Change color gammut to feed the frame into the network
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 output = run_inference_for_single_image(frame, sess, 
    #                     detection_graph)
    #                 output = discriminate_class(output, 
    #                     classes_to_detect, category_index)
    #                 processed_image = visualize_results(frame, output, 
    #                     category_index)

    #                 # Display the image with the detections in the Streamlit app
    #                 img_placeholder.image(processed_image)
                    
    #                 #cv2.imshow('Video', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    #                 # if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 #     break
    
    # except KeyboardInterrupt:   
    #     pass

    # print("Ending resources")
    # st.text("Camera not detected")
    # cv2.destroyAllWindows()
    # video_thread.stop()
    # sys.exit()


#---------------------------------------------------------------------------------------------------------------------------------------
"""
def main():
    #with st.expander("Les objets routiers:"):
        #st.markdown( '<p style="font-size: 20px;">personne, voiture, vélo, moto, camion, bus, signe stop, feu de circulation </p>', unsafe_allow_html= True)
        #st.markdown('<p style = "font-size : 20px; color : white;">This app was built using Streamlit, Detectron2 and OpenCv to demonstrate <strong>Object Detection</strong> in both videos (pre-recorded) and images.</p>', unsafe_allow_html=True)
        
    
    

    option = st.selectbox(
     'Avec quel type de fichier souhaitez-vous travailler ?',('Images', 'Videos','Webcam'))

    #st.write('You selected:', option)
    if option == "Images":
        st.title('Détection d\'objets pour les images')"""
        #st.subheader("""Une image est générée avec des cadres de délimitation créés autour des objets routiers de l'image.""")
        #func_1('objectDetection')
    #elif option =="Videos":
        #st.title('Détection d\'objets pour les vidéos')
        #st.subheader("""Une vidéo est générée avec des cadres de délimitation créés autour des objets routiers de la vidéo.""")
        #func_2('objectDetection') 
    #else:
       #st.title('Détection en temps réel')
       #if st.button('Commencer'):
        #princip()

        
#------------------------------------------------------------------------



    
def load_view_D2():

    st.title('Détection avec Detectron2')      
    main()



  