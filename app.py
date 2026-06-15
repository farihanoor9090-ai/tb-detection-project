import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import math
from PIL import Image
import tempfile

# Load model
model = YOLO("exp-2.pt")

st.title("AI-Based Tuberculosis Detection + Optical Density + Dose Estimation")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    image.save(temp_file.name)

    image_path = temp_file.name

    st.subheader("TB Detection")

    results = model(image_path)

    detected = False

    for result in results:

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            detected = True

            for box in boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                st.success(
                    f"TB Detected: {model.names[cls]}"
                )

                st.write(
                    f"Confidence: {round(conf*100,2)}%"
                )

    if not detected:
        st.warning("No TB detected")

    img = cv2.imread(image_path, 0)

    mean_pixel = np.mean(img)

    mean_pixel = max(mean_pixel, 1)

    I0 = 255

    od = math.log10(I0 / mean_pixel)

    dose = od * 1.5

    st.subheader("Optical Density")

    st.write(round(od, 3))

    st.subheader("Estimated Dose")

    st.write(
        f"{round(dose,3)} mGy"
    )