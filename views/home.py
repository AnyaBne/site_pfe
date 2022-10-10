import streamlit as st
from PIL import Image
from views import detect_D2
import detect_yolo

from PIL import Image



def load_view():
    
   
    st.title("Bienvenue sur notre application Web de détection d'objets routiers")
    img1 = Image.open("ph.png")
    st.image(img1, caption='', width=750)

    #img2= Image.open("C:/Users/DELL Latitude 7270/Desktop/Code_Site_Web/views/pres.png")
    #img3= Image.open("C:/Users/DELL Latitude 7270/Desktop/Code_Site_Web/views/usthb.png")

    #st.image(img1, caption='')
    #col1, col2 = st.columns(2)

    #with col1:
     #st.image(img1, caption='', width=900)

    #with col2:
     #st.image(img2, caption='',width=600)

    #with col3:
     #st.image(img3, caption='')

 
    st.subheader('Dans la barre de navigation, vous avez le choix entre deux modèles de détection: Detectron2 ou YOLOv5')
   
    st.subheader("Pour plus d\'information sur les modèles veuillez consulter la partie A propos")
  


   
   
   
    #btn1 = st.button("Detectron2")
    #btn2 = st.button("Yolo V5")

    #if btn1:
      #st.session_state.runpage = "Detection avec Detectron2"
      #st.session_state.runpage()
      #st.experimental_rerun()

    #if btn2:
      #st.session_state.runpage = "Detection avec Yolo v5"
      #st.session_state.runpage()
      #st.experimental_rerun()   