import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model as keras_load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import numpy as np
import cv2
import cvlib as cv

st.set_page_config('Gender Detection', '👦👧', 'wide', 'auto')

if 'image_path' not in st.session_state:
    st.session_state['image_path'] = ''


@st.cache_resource
def load_model():
    model = keras_load_model('gender_detection.model')
    return model


classes = ['man', 'woman']


def process_frame(img):
    face, confidence = cv.detect_face(img)
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_crop = np.copy(img[startY:endY, startX:endX])
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = load_model().predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)
        print(f"Detected face at {(startX, startY, endX, endY)}, label: {label}")
    return img


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_frame(img)
        return img


main_container = st.container()
with main_container:
    st.title('Gender Detection')
    tab1, tab2 = st.tabs(['Upload Image', 'Open Camera'])
    with tab1:
        uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            processed_image = process_frame(image)
            st.image(processed_image, channels="BGR", width=700)
    with tab2:
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
