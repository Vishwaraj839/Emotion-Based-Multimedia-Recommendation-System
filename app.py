from flask import Flask, render_template, request, Response, redirect, url_for
import sqlite3
import numpy as np
import cv2
import hashlib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from werkzeug.utils import secure_filename
import warnings
from collections import deque
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
import random
warnings.filterwarnings('ignore')


# Initialize Flask app
app = Flask(__name__)

# Global variable to store the current emotion
emotion_result = None

# Spotify API setup
SPOTIFY_CLIENT_ID = '525c83802b604943b43c1a62f76bf13c'
SPOTIFY_CLIENT_SECRET = '64f7569847654d06b79a20937eee694f'
spotify = Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                        client_secret=SPOTIFY_CLIENT_SECRET))

# YouTube API setup
YOUTUBE_API_KEY = 'AIzaSyBBC9Syq_gH0KtgK8dY0VPWTG_0-cxJ-aQ'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Emotion-to-Music mapping
emotion_music = {
    "Angry": ["energetic", "metal"],
    "Disgusted": ["motivational"],
    "Fearful": ["relaxing", "calming"],
    "Happy": ["upbeat", "pop"],
    "Neutral": ["lo-fi", "chill"],
    "Sad": ["calm", "instrumental"],
    "Surprised": ["party", "fast-paced"]
}


# Fetch songs based on emotion from Spotify
def fetch_spotify_songs(emotion):
    genres = emotion_music.get(emotion, ["chill"])
    query = random.choice(genres)
    results = spotify.search(q=query, type='track', limit=5)
    return [{"name": track["name"], "url": track["external_urls"]["spotify"]} for track in results['tracks']['items']]


# Fetch songs based on emotion from YouTube
def fetch_youtube_songs(emotion):
    genres = emotion_music.get(emotion, ["chill"])
    query = random.choice(genres)
    request = youtube.search().list(q=query, part="snippet", maxResults=5)
    response = request.execute()
    return [{"name": item["snippet"]["title"], "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"}
            for item in response["items"] if item["id"]["kind"] == "youtube#video"]


def emotion_detection():
    global emotion_result
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('D:/music_emotion/model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start the webcam feed
    cap = cv2.VideoCapture(0)

    # Buffer to store the last 10 emotions
    emotion_buffer = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion_buffer.append(emotion_dict[maxindex])  # Append to buffer

            # Calculate the most frequent emotion in the buffer
            emotion_result = max(set(emotion_buffer), key=emotion_buffer.count)

            # Display the current emotion
            cv2.putText(frame, emotion_result, (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              password_hash TEXT,
              email VARCHAR,
              phone_no INTEGER,
              R_address VARCHAR(255),
              gender VARCHAR,
              age INTEGER,
              dob DATE)''')

    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def login():
    global loggedin_user
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            
            # Hash the password for comparison
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
            user = c.fetchone()
            conn.close()
            
            if user:
                # Successful login, redirect to home page
                loggedin_user = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[3],
                    'phone_no': user[4],
                    'address': user[5],
                    'gender': user[6],
                    'age': user[7],
                    'dob': user[8]
                }
                return redirect('/index')
            else:
                # Invalid credentials, render login page with error message
                return render_template('login1.html', error='Invalid username or password')
        
        except Exception as e:
            # Handle any exceptions
            return render_template('error.html', message="An error occurred during login. Please try again later.")

    # If it's a GET request, render the login page
    return render_template('login1.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            email = request.form['email']
            phone = request.form['phone_no']
            R_address = request.form['R_address']
            gender = request.form['gender']
            age = request.form['age']
            dob = request.form['dob']

            # Hash the password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, password_hash, email, phone, R_address, gender, age, dob))
            conn.commit()
            conn.close()

            return "Registration successful!", 200

        except Exception as e:
            print("Error during registration:", e)
            return "An error occurred during registration. Please try again later.", 500

    return render_template('register1.html')

# Home route1
@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/emotion', methods=['GET'])
def emotion():
    spotify_songs = fetch_spotify_songs(emotion_result)
    youtube_songs = fetch_youtube_songs(emotion_result)
    return render_template('emotion.html',spotify_songs=spotify_songs, youtube_songs=youtube_songs)

@app.route('/emotion_video', methods=['GET'])
def emotion_video():
    return Response(emotion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
``