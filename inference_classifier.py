import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import tkinter as tk
from tkinter import Button, Label
from threading import Thread
from PIL import Image, ImageTk

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary for alphabets and blank space
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = " "  # Assign space for blank detection

# Variables for sentence formation and timer
sentence = ""
current_letter = ""
last_detected_time = time.time()
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def play_audio():
    speak(sentence)

def clear_last_letter():
    """Clear the last character from the sentence."""
    global sentence
    if sentence:
        sentence = sentence[:-1]

# Create Tkinter window for GUI
root = tk.Tk()
root.title("Sign Language Recognition")

# Add a label for displaying the real-time video frame
video_label = Label(root)
video_label.pack()

# 'Play' button to speak the sentence
play_button = Button(root, text="Play", command=lambda: Thread(target=play_audio).start())
play_button.pack(pady=10)

# 'Clear' button to erase the last character
clear_button = Button(root, text="Clear Last Letter", command=clear_last_letter)
clear_button.pack(pady=10)

# Add a label to display the detected sentence
sentence_label = Label(root, text="Sentence: ", font=('Helvetica', 12), wraplength=400)
sentence_label.pack(pady=10)

# Add a label to display the currently detected character
current_letter_label = Label(root, text="Current Letter: ", font=('Helvetica', 12), fg="blue")
current_letter_label.pack(pady=10)

def update_frame():
    global sentence, current_letter, last_detected_time

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            x_min = min(x_)
            y_min = min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - x_min)
                data_aux.append(y - y_min)
    else:
        # If no hand is detected, classify as blank
        data_aux = [0] * 84

    current_time = time.time()
    if current_time - last_detected_time >= 3:  # Update every 3 seconds
        # Ensure feature length matches model input
        expected_length = 84  # Adjust based on the dataset
        if len(data_aux) < expected_length:
            data_aux += [0] * (expected_length - len(data_aux))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Update the current detected character
        current_letter = predicted_character
        current_letter_label.config(text=f"Current Letter: {current_letter}")

        # Append the detected character to the sentence after preview
        sentence += predicted_character
        last_detected_time = current_time

    # Convert the frame to an image that Tkinter can display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = ImageTk.PhotoImage(image)

    # Update the video label with the new frame
    video_label.config(image=image)
    video_label.image = image

    # Update the sentence label with the current sentence
    sentence_label.config(text=f"Sentence: {sentence}")

    # Continue updating the frame every 10ms
    root.after(10, update_frame)

# Start updating frames in the Tkinter window
update_frame()

# Display the Tkinter window
root.geometry("600x700")
root.protocol("WM_DELETE_WINDOW", root.quit)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
