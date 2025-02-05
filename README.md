American-Sign-Language-Deteaction-and-TextToSpeech-Concersion-Towards-Inclusivity




Overview:

This project addresses the communication gap between speech-impaired and speaking individuals by providing a real-time American Sign Language (ASL) recognition system. The system translates ASL gestures into text and converts them into speech, enabling seamless interaction.




Features:

Real-time ASL gesture recognition using a webcam

Hand landmark detection using MediaPipe

Gesture classification using a Random Forest model

Text conversion and synchronized audio output using pyttsx3

User-friendly graphical interface for smooth interaction




Technologies Used:

Programming Language: Python

Libraries: OpenCV, NumPy,tkinter, MediaPipe, pyttsx3

Machine Learning Model: Random Forest Classifier

Development Environment: Spyder (Python 3.9, 64-bit)




Installation:

Prerequisites:

Ensure you have Python 3.9 installed. Then, install the required dependencies:

pip install opencv-python numpy tensorflow keras mediapipe pyttsx3

Clone the Repository:

git clone https://github.com/your-username/ASL-Gesture-Recognition.git
cd ASL-Gesture-Recognition




Usage:

1)Collect dataset in real-time:

python collect_images.py

This will capture hand gesture images via webcam.


2)Create dataset (pickle file):

python create_dataset.py

This will process and store the collected data in a pickle file.


3)Train the Random Forest model:

python train_classifier.py

This will train the model and save it as model.p.


4)Run real-time detection and text-to-speech conversion

python inference_classifier.py

The webcam will activate and capture hand gestures.

The system detects and classifies gestures in real time.

Recognized gestures are converted into text and displayed.

Press the 'Speak' button to hear the text output.


You can directly run python inference_classifier.py as the already trained model is uploaded.





Dataset:

The system is trained on a custom ASL dataset containing 26 folders with folder 0-25 for A-Z letters and 26th folder with image representing spaces. Each image is preprocessed and used for model training.




Model Training:

The dataset features 21 hand landmarks (x, y coordinates) extracted using MediaPipe.

Normalization is applied to improve model accuracy.

A Random Forest Classifier is trained on the extracted features and saved as model.p.




Results:

High accuracy in recognizing ASL alphabet gestures with 98%

Low latency for real-time gesture detection and conversion

Effective speech output for recognized gestures




Limitations:

Recognition limited to ASL alphabets (no words or sentences)

Accuracy may be affected by inconsistent lighting conditions

Requires further expansion to include dynamic gestures




Future Enhancements:

Extend recognition to ASL words and sentences

Improve robustness under varying lighting conditions

Integrate facial expression recognition for enhanced communication




Contributors:

Hitha B Mendon - Mangalore Institute of Technology & Engineering
Savi Sanjiv - Mangalore Institute of Technology & Engineering
Priyanka U S - Mangalore Institute of Technology & Engineering
Shetty Samarth Gunapal - Mangalore Institute of Technology & Engineering
