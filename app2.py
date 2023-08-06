import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import requests
from mtcnn import MTCNN


def detect_faces(our_image):
    face_detector = MTCNN()
    img_rgb = cv2.cvtColor(our_image, cv2.COLOR_BGR2RGB) #mtcnn expects RGB but OpenCV read BGR
    detections = face_detector.detect_faces(img_rgb)
    detection = detections[0]
    
def draw_bounding_boxes(our_image, detection):
    for detections in detection:
        x1, y1, w, h = detection['box']
        cv2.rectangle(our_image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)


    #x, y, w, h = detection["box"]
    #detected_face = img[int(y):int(y+h), int(x):int(x+w)]
    #return detected_face



def mark_key_point(our_image, keypoint):
    cv2.circle(our_image, (keypoint), 1, (0,255,0), 2)

# draw bounding box around the detected face and mark facial keypoints
def draw_bounding_boxes(our_image, detection):
    mark_key_point(our_image, detection[0]['keypoints']['left_eye'])
    mark_key_point(our_image, detection[0]['keypoints']['right_eye'])
    mark_key_point(our_image, detection[0]['keypoints']['nose'])
    mark_key_point(our_image, detection[0]['keypoints']['mouth_left'])
    mark_key_point(our_image, detection[0]['keypoints']['mouth_right'])



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


plt.figure(figsize=(10,10))
plt.imshow(image_file)
plt.xticks([])
plt.yticks([])
plt.show()



if __name__ == '__main__':
    main()
