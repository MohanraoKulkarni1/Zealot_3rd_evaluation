import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2
from threading import Thread

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Function Detection")
        self.root.geometry("400x200")

        self.current_function = None

        self.create_widgets()

    def create_widgets(self):
        self.eye_button = tk.Button(self.root, text="Eye Detection", command=self.run_eye_detection)
        self.eye_button.pack(pady=10)

        self.hand_button = tk.Button(self.root, text="Hand Detection", command=self.run_hand_detection)
        self.hand_button.pack(pady=10)

        self.lip_button = tk.Button(self.root, text="Lip Detection", command=self.run_lip_detection)
        self.lip_button.pack(pady=10)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit)
        self.quit_button.pack(pady=10)

    def run_eye_detection(self):
        if self.current_function:
            self.current_function.join()
        self.current_function = Thread(target=self.eye_detection())
        self.current_function.start()

    def run_hand_detection(self):
        if self.current_function:
            self.current_function.join()
        self.current_function = Thread(target=self.hand_detection())
        self.current_function.start()

    def run_lip_detection(self):
        if self.current_function:
            self.current_function.join()
        self.current_function = Thread(target=self.lip_detection())
        self.current_function.start()

    def quit(self):
        if self.current_function:
            self.current_function.join()
        self.root.quit()

    def eye_detection(self):
        #---nightvision based on time and drowsiness detection with eye and sanding message and email ---

        import cv2
        import dlib
        from imutils import face_utils
        from scipy.spatial import distance
        from pygame import mixer
        import imutils
        from twilio.rest import Client
        import smtplib
        import datetime
        from email.message import EmailMessage
        
        def sms(body1, from1, to1):
            account_ssid='AC9eb13e8523a6c0d87e6e8b8248ea1c38'
            auth_token='4dad115dd4fd0b57d306f22e519e5dd0'
            client=Client(account_ssid,auth_token)
            message=client.messages\
            .create(
                body=body1,
                from_=from1,
                to=to1
            )
        
        def email_alert(subject, body, to):
            msg=EmailMessage()
            msg.set_content(body)
            
            msg['subject']=subject
            msg['to']=to
            
            user="wave01072024@gmail.com"
            msg['from']=user
            password="tgdjbqsityxwzuzs"
            
            server=smtplib.SMTP("smtp.gmail.com",587)
            server.starttls()
            server.login(user,password)
            server.send_message(msg)
            
            server.quit()
        
        mixer.init()
        mixer.music.load("music.wav")
        
        def eye_aspect_ratio(eye):
            A = distance.euclidean(eye[1], eye[5])
            B = distance.euclidean(eye[2], eye[4])
            C = distance.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear
        
        thresh = 0.25
        frame_check = 20
        
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        
        cap = cv2.VideoCapture(0)
        flag = 0
        numk = 0
        
        # Function to apply night vision effect
        def apply_night_vision(frame):
            # Split the frame into its color channels
            b, g, r = cv2.split(frame)
        
            # Enhance the green channel
            g = cv2.equalizeHist(g)
        
            # Merge back the channels
            night_vision_frame = cv2.merge((b, g, r))
        
            return night_vision_frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Check current time
            current_time = datetime.datetime.now().time()
            if current_time < datetime.time(6, 0) or current_time > datetime.time(19, 0):
                # Apply night vision effect to the frame
                frame = apply_night_vision(frame)
            else:
                # Use normal camera settings
                pass
            
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                
                eyeHull = cv2.convexHull(leftEye)
                cv2.drawContours(frame, [eyeHull], -1, (0, 255, 0), 1)
                eyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [eyeHull], -1, (0, 255, 0), 1)
                
                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        numk = numk + 1
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if numk == 11:
                            email_alert("hey", "drowsiness has been detected from mohan, please contact the driver", "suppuratnakar@gmail.com")
                            sms('Suspicious activities are detected please contact the driver', '+12675926853', '+919731247737')
                        if not mixer.music.get_busy():
                            mixer.music.play()
                else:
                    flag = 0
                    if mixer.music.get_busy():
                        mixer.music.stop()
            
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def hand_detection(self):
        #multiple hand detection 
        import cv2
        import mediapipe as mp
        import pygame
        import time
        
        # Initialize hand detection model
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        
        # Initialize pygame for music playback
        pygame.mixer.init()
        music_playing = False
        start_time = None
        
        # Define function to find hands and draw landmarks
        def findHands(image, draw=True):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
        
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if draw:
                        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            return image, results
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            success, image = cap.read()
            if not success:
                break
        
            image, results = findHands(image)
        
            # Check if hands were detected
            if results.multi_hand_landmarks:
                cv2.putText(image, "Hands Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if len(results.multi_hand_landmarks) == 2:
                    if start_time is None:
                        start_time = time.time()
                    elif (time.time() - start_time) >= 1 and not music_playing:
                        pygame.mixer.music.load("music.wav")
                        pygame.mixer.music.play()
                        music_playing = True
                else:
                    start_time = None
            else:
                cv2.putText(image, "Hand Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if music_playing:
                    pygame.mixer.music.stop()
                    music_playing = False
                    start_time = None
        
            # Display image
            cv2.imshow("Hand Detection", image)
        
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


    def lip_detection(self):
        #---lips detection with night vision based on current time
        import cv2
        import dlib
        import numpy as np
        import pygame
        import os
        import datetime
        
        # Load pre-trained face and landmark detector models from dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Function to calculate lip distance (vertical distance between top and bottom lip landmarks)
        def lip_distance(shape):
            top_lip = shape[50:53]
            top_lip = np.concatenate((top_lip, shape[61:64]))
        
            bottom_lip = shape[56:59]
            bottom_lip = np.concatenate((bottom_lip, shape[65:68]))
        
            top_mean = np.mean(top_lip, axis=0)
            bottom_mean = np.mean(bottom_lip, axis=0)
        
            distance = abs(top_mean[1] - bottom_mean[1])
            return distance
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Variables for drowsiness detection
        lip_distance_threshold = 18
        frame_counter = 0
        drowsy = False
        
        # Initialize Pygame mixer for audio playback
        pygame.mixer.init()
        pygame.mixer.music.load('music.wav')  # Load the music file
        
        # Function to apply night vision effect
        def apply_night_vision(frame):
            # Split the frame into its color channels
            b, g, r = cv2.split(frame)
        
            # Enhance the green channel
            g = cv2.equalizeHist(g)
        
            # Merge back the channels
            night_vision_frame = cv2.merge((b, g, r))
        
            return night_vision_frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Check current time
            current_time = datetime.datetime.now().time()
            if current_time < datetime.time(6, 0) or current_time > datetime.time(19, 0):
                # Apply night vision effect to the frame
                frame = apply_night_vision(frame)
            else:
                # Use normal camera settings
                pass
        
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Detect faces in the grayscale frame
            faces = detector(gray)
        
            for face in faces:
                # Predict facial landmarks
                landmarks = predictor(gray, face)
        
                # Convert landmarks to numpy array
                shape = np.zeros((68, 2), dtype=int)
                for i in range(0, 68):
                    shape[i] = (landmarks.part(i).x, landmarks.part(i).y)
        
                # Calculate lip distance
                lip_dist = lip_distance(shape)
        
                # Draw landmarks on face
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
                # Check for drowsiness
                if lip_dist > lip_distance_threshold:
                    frame_counter += 1
                    if frame_counter > 10:  # If lip movement is sustained for some frames
                        if not drowsy:
                            pygame.mixer.music.play(-1)  # Play music in a loop
                            drowsy = True
                else:
                    frame_counter = 0
                    if drowsy:
                        pygame.mixer.music.stop()  # Stop playing music
                        drowsy = False
        
            # Display the frame
            cv2.imshow("Frame", frame)
        
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and destroy any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

       

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
