import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # Import the text-to-speech library

# Load the pre-trained model
model_dict = pickle.load(open(r"D:\app\sign-language-detector-python-master\model.p", 'rb'))
model = model_dict['model']

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary for prediction
labels_dict = {0: '1', 1: '2', 2: 'Good', 3: 'Morning', 4: 'A', 5: 'X'}

# Initialize pyttsx3 engine for text-to-speech
engine = pyttsx3.init()

# Variables for tracking prediction stability
last_prediction = None
stable_count = 0
stable_threshold = 10  # Number of frames to wait for stable gesture

if not cap.isOpened():
    print("Error: Camera not opened.")
else:
    print("Camera opened successfully.")

while True:
    ret, frame = cap.read()

    if not ret:  # Check if frame is captured successfully
        print("Failed to grab frame")
        break

    # Flip the frame horizontally to fix the mirror effect
    frame = cv2.flip(frame, 1)

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):  # Process all landmarks
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(21):  # Compute relative coordinates
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure `data_aux` has exactly 100 features
        required_features = 100
        if len(data_aux) < required_features:
            # Pad with zeros if fewer features
            data_aux.extend([0] * (required_features - len(data_aux)))
        elif len(data_aux) > required_features:
            # Truncate to required length
            data_aux = data_aux[:required_features]

        # Prepare the bounding box for drawing
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make predictions using the model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw the prediction and bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Check if the prediction has changed
        if predicted_character == last_prediction:
            # Increase the stability count if the gesture is consistent
            stable_count += 1
        else:
            # Reset if the gesture changes
            last_prediction = predicted_character
            stable_count = 0

        # Only speak if the gesture is stable for enough frames
        if stable_count >= stable_threshold:
            engine.say(predicted_character)
            engine.runAndWait()
            stable_count = 0  # Reset the counter after speaking

    # Show the frame with annotations
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
        break

cap.release()
cv2.destroyAllWindows()
