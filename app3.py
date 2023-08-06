import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import matplotlib as plt
from mtcnn import MTCNN
from detection import detect_face,draw_bounding_boxes,mark_key_point


def main():

    st.set_page_config(page_title="Face Detection" )
    
    st.title("Face Detection App")

    image_file = st.text_input('Enter valid url')
    
    if image_file:
        response = requests.get(image_file)
        img = Image.open(BytesIO(response.content))
        st.image(img,width=500, caption='Input Image')

    if st.button("Verify"):
        result_img, result_faces = detect_face(img)
        st.image(result_img, caption='Output Image', width=500)
        st.success("Found {} faces".format(len(result_faces)))


if __name__ == '__main__':
    main()