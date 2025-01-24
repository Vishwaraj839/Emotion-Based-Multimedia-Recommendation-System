# Emotion-Based-Multimedia-Recommendation-System

Emotion Detection via Webcam:

Uses a Convolutional Neural Network (CNN) model to predict emotions based on live webcam feed.
The result is smoothed using a buffer (deque) to handle fluctuations in predictions.
Integration with Spotify and YouTube APIs:

Fetches playlists or videos based on the detected emotion using predefined emotion-to-genre mappings.
User Authentication:

Implements a SQLite database to manage user registration and login securely with hashed passwords.
Routes Overview:

/: Login page.
/register: Registration page.
/index: Home page after successful login.
/emotion: Displays Spotify and YouTube recommendations based on detected emotions.
/emotion_video: Streams the live webcam feed with emotion detection.
/logout: Logs out the user and redirects to the login page.
