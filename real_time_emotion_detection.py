import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
model = load_model('emotion_recognition_model2.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(image):
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi_gray)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]

        # Draw a rectangle around the face and display the emotion label
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image_np, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Convert the image back to PIL format
    return Image.fromarray(image_np)

# Streamlit UI
st.title('Emotion Recognition')
st.write('Upload an image to predict the emotion.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    output_image = predict_emotion(image.copy())
    st.image(output_image, caption='Processed Image with Emotion Label', use_column_width=True)
