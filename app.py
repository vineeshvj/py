import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import requests



face_cascade = cv2.CascadeClassifier(r'C:\Users\MAHE\Desktop\Facedetection\haarcascade_frontalface_default.xml')


#HAAR
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=4,minSize=(34, 35), flags=cv2.CASCADE_SCALE_IMAGE)
    #faces = face_cascade.detectMultiScale(gray, 1.1,4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    return img, faces


def main():

    st.set_page_config(page_title="Face Detection" )
    
    st.title("Face Detection App")

    image_file = st.text_input('Enter valid url')
    
    if image_file:
        response = requests.get(image_file)
        img = Image.open(BytesIO(response.content))
        st.image(img,width=500, caption='Input Image')

    if st.button("Verify"):
        result_img, result_faces = detect_faces(img)
        st.image(result_img, caption='Output Image', width=500)
        st.success("Found {} faces".format(len(result_faces)))


if __name__ == '__main__':
    main()