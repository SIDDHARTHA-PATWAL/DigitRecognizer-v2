import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the Keras model
try:
    model = tf.keras.models.load_model("handwritten_model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

def predict_digit(image):
    # Preprocess the image: Convert to grayscale and resize to 28x28
    numpydata = np.array(image)
    numpydata = numpydata.reshape(1, 28 * 28)
    numpydata = 255 - numpydata  # Invert the image colors

    # Predict using the model
    prediction = model.predict(numpydata.reshape(1, 784))
    prediction_p = tf.nn.softmax(prediction).numpy()

    predicted_digit = np.argmax(prediction_p)
    confidence = prediction_p[0][predicted_digit]
    return predicted_digit, confidence

def main():
    st.title("Digit Recognizer - Draw or Upload an Image of a Digit")

    # Option to upload an image or draw on canvas
    option = st.selectbox("Choose input method", ("Draw on Canvas", "Upload an Image"))

    if option == "Upload an Image":
        # File uploader for image
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Open the uploaded image
            image = Image.open(uploaded_file).convert('L')

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Predict digit on button click
            if st.button("Predict Digit"):
                predicted_digit, confidence = predict_digit(image)
                st.write(f"Predicted Digit: **{predicted_digit}**")
                st.write(f"Confidence: **{confidence:.2%}**")

    elif option == "Draw on Canvas":
        # Add a button to clear the canvas
        clear_canvas = st.button("Clear Canvas")

        # Create a canvas component with an option to clear it
        canvas_result = st_canvas(
            fill_color="white",        # Background color of the canvas
            stroke_width=10,           # Thickness of the brush
            stroke_color="black",      # Brush color
            background_color="white",  # Canvas background color
            width=280,                 # Canvas width
            height=280,                # Canvas height
            drawing_mode="freedraw",   # Free drawing mode
            key="canvas" if not clear_canvas else "reset_canvas"  # Reset canvas when Clear is pressed
        )

        # Predict digit from canvas drawing
        if st.button("Predict Digit from Canvas"):
            if canvas_result.image_data is not None:
                # Convert the drawing to an image and preprocess it
                img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
                img = img.convert("L").resize((28, 28))  # Convert to grayscale and resize to 28x28
                # img = img.convert("RGB").convert("L").resize((28, 28), Image.LANCZOS)
                
                # Display the drawn image
                st.image(img, caption="Your Drawing", use_container_width=True)

                # Predict the digit
                predicted_digit, confidence = predict_digit(img)
                st.write(f"Predicted Digit: **{predicted_digit}**")
                st.write(f"Confidence: **{confidence:.2%}**")
            else:
                st.warning("Please draw a digit on the canvas.")

if __name__ == "__main__":
    main()
