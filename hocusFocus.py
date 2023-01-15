import cv2
from playsound import playsound
import time

# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture object
cap = cv2.VideoCapture(0)

# Timer to keep track of time since last face detection
timer = time.time()

# Flag to check if audio is currently playing
audio_playing = False

while True:
    # Get a frame from the video capture object
    ret, frame = cap.read()
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # No faces are detected in the frame
    if len(faces) == 0:
        # Check if 5 seconds have passed since last face detection; indicates user is distracted
        if time.time() - timer > 5: 
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Play an audio file until face comes back into frame
            playsound('Sony_Xperia_Toques_Notification_Sound_Effect.mp3', block=False)
            # Give a gap between occurences of the alarm sound to ensure the sound completely finishes before repeating it
            time.sleep(1)

    else:
        # Update the timer
        timer = time.time()
        audio_playing = False
        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get the grayscale image of the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

    # Show the frame
    cv2.imshow('HocusFocus', frame)
    cv2.setWindowProperty("HocusFocus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
