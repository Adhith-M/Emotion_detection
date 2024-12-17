import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model # type: ignore

# Load the facial emotion recognition model
model = load_model('model.h5')

# List of emotions based on the model's training
emotions =  ['Happy', 'Sad', 'Neutral']


# Initialize the GUI
app = tk.Tk()
app.geometry('800x600')
app.title('Facial Emotion Recognition')
app.configure(background='#CDCDCD')

# Labels for emotion detection result
result_label = Label(app, background="#CDCDCD", font=('arial', 15, "bold"))
image_label = Label(app)

# Function to process and predict emotion from the uploaded image
def detect_emotion(file_path):
    try:
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        image = image.resize((48, 48))  # Resize to model's input size
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize pixel values

        # Predict the emotion
        predictions = model.predict(image)
        emotion = emotions[np.argmax(predictions)]

        # Update the result label
        result_label.configure(foreground="#011638", text=f'Emotion: {emotion}')
    except Exception as e:
        result_label.configure(foreground="red", text="Error in detection")
        print(e)

# Function to handle image upload
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((app.winfo_width() / 2.25), (app.winfo_height() / 2.25)))
        img = ImageTk.PhotoImage(uploaded)

        image_label.configure(image=img)
        image_label.image = img
        result_label.configure(text='')

        # Create the detect button dynamically
        detect_button = Button(app, text="Detect Emotion", command=lambda: detect_emotion(file_path), padx=10, pady=5)
        detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        detect_button.place(relx=0.79, rely=0.46)
    except Exception as e:
        print(e)

# Upload button
upload_button = Button(app, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload_button.pack(side='bottom', pady=50)

# Placeholders for image and result
image_label.pack(side='bottom', expand=True)
result_label.pack(side="bottom", expand=True)

# Heading label
heading = Label(app, text="Facial Emotion Recognition", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start the GUI
app.mainloop()
