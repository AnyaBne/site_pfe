import streamlit as st

def load_view():    
  st.title('Detectron2 ou YOLOv5 ?')
  st.subheader("Qu'est-ce que Detectron2 ?")
  st.write("Detectron2 est la bibliothèque de nouvelle génération de Facebook AI Research qui fournit des algorithmes de détection et de segmentation de pointe. C'est le successeur de Detectron. Il prend en charge un certain nombre de projets de recherche en vision par ordinateur et d'applications de production sur Facebook.")
  st.subheader("Qu'est-ce que YOLOv5 ?")
  st.write("YOLO (You Only Look Once) est un algorithme de détection d'objets en une étape. Les détecteurs d'objets à une étape comme YOLO analysent une image en un seul passage (d’où son nom) et génèrent plusieurs prédictions d'emplacement et de classification d'objets. En conséquence.YOLO est l'un des algorithmes de détection d'objets les plus connus en raison de sa rapidité et de sa précision.")
  st.write("   ")
  st.write("Les deux modèles ont été entrainé sur la dataset MS COCO customisé, en sélectionnant seulement les objets routiers")
  st.write("   ")
  st.write("YOLOv5 est meilleur en terme de rapidité en temps d'exécution que ça soit pour la détection sur des images ou sur une video, mais aussi il est beaucoup plus fluide dans la detection en temps réel. YOLOv5 possède également une meilleure précision")
