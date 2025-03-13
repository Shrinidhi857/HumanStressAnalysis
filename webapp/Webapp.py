"""_summary_
    This Program act as backend and controlling subprocesses.
    
"""
from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from joblib import load  # For loading the ML model
import os
from datetime import timedelta
import subprocess
import json
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import threading
import matplotlib
import soundfile as sf
from tensorflow.keras.models import load_model
import pickle
import librosa
from pydub import AudioSegment
import threading

TASK_STATUS = {"completed": False}
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)



# Route: Initialize Database
@app.route('/init_db')
def init_db():
    """Creates all necessary tables."""
    with app.app_context():
        db.create_all()
    return "Database initialized and tables created!"

# Route: Home
@app.route('/')
def home():
    username = session.get('username')
    return render_template('InterfaceIndex.html', username=username)

@app.route("/view_users")
def view_users():
    # Fetch all users using SQLAlchemy
    users = User.query.all()  # Retrieve all users from the database
    return render_template("view_users.html", users=users)

# Route: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = user.username
            session.permanent = True
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials. Please try again.", "error")
    return render_template('login.html')

# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not username or not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose a different one.", "error")
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash("Email already exists. Please use a different one.", "error")
            return render_template('register.html')

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

# Route: Logout
@app.route('/logout', methods=['POST'])
def logout():
    if 'username' in session:
        entered_password = request.form['password']
        user = User.query.filter_by(username=session.get('username')).first()

        if user and check_password_hash(user.password, entered_password):
            session.pop('username', None)
            flash('You have been logged out successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Incorrect password. Please try again.', 'error')
    return redirect(url_for('home'))

# Route: Take Test
@app.route('/take_test', methods=['GET', 'POST'])
def take_test():
    if 'username' not in session:
        flash('You must be logged in to take the test.', 'error')
        return redirect(url_for('login'))
    
    return render_template('take_test.html')

UPLOAD_FOLDER = 'C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

import threading

@app.route('/submit_test', methods=['POST'])
def submit_test():
    try:
        # Save uploaded files
        for key, file in request.files.items():
            if file:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                app.logger.info(f"File saved: {file_path}")
            else :
                app.logger.info(f"File saved: {file_path}")
        print(f"file uploaded {request.files.items()}")
        metadata = request.form.get("metadata")
        if metadata:
            metadata = json.loads(metadata)
            app.logger.info(f"Metadata received: {metadata}")
                               
        run_background_tasks()
        return result_page(),202

    except Exception as e:
        app.logger.error(f"Error in submit_test: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def check_status():
    global TASK_STATUS
    if TASK_STATUS["completed"]:
        return jsonify({"status": "completed"}), 200
    else:
        return jsonify({"status": "in_progress"}), 202
    
        
@app.route('/result',methods=['GET']) 
def result_page():
    chart_details = analyze_emotions()
    return render_template('result.html', 
                           emotion_counts=chart_details["counts"], 
                           line_chart_path=chart_details['line_chart_path'],
                          
                           
                           )
 

       
    

    
def run_background_tasks():
    global TASK_STATUS
    try:
        file_path = "webapp\\audiofile-step1\\merged_audio.wav"
        folder_2min = "webapp\\audiofile-step3-2"
        folder_3sec = "webapp\\audiofile-step3-3"
        
        #audio_adder()
        split_audio_to_folders(file_path, folder_2min, folder_3sec)
        class_predictor()
        
        
        
    except Exception as e:
        print(f"Error running scripts: {e}")
        TASK_STATUS["completed"] = False


def audio_adder():
    remove_files_in_folder('webapp\\audiofile-step1')
    audio_dir = UPLOAD_FOLDER
    output_dir = 'webapp\\audiofile-step1' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    audio_files.sort()
    merged_audio = AudioSegment.empty()
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(".webm"):
            file_path = os.path.join(audio_dir, audio_file)
            audio = AudioSegment.from_file(file_path, format="webm")
            merged_audio += audio
    output_file_path = os.path.join(output_dir, 'merged_audio.wav')
    merged_audio.export(output_file_path, format='wav')

    print(f'Merged audio has been saved to: {output_file_path}')
    remove_files_in_folder("uploads")


def split_audio_to_folders(file_path, folder_2min, folder_3sec):
    remove_files_in_folder(folder_2min)
    remove_files_in_folder(folder_3sec)
    try:
        audio = AudioSegment.from_file(file_path, format="wav")
        os.makedirs(folder_2min, exist_ok=True)
        os.makedirs(folder_3sec, exist_ok=True)

        chunk_duration_2min = 2 * 60 * 1000
        for i in range(0, len(audio), chunk_duration_2min):
            chunk = audio[i:i + chunk_duration_2min]
            chunk_filename = f"chunk_{i // chunk_duration_2min + 1}.wav"
            chunk.export(os.path.join(folder_2min, chunk_filename), format="wav")

        chunk_duration_3sec = 3 * 1000
        for i in range(0, len(audio), chunk_duration_3sec):
            chunk = audio[i:i + chunk_duration_3sec]
            chunk_filename = f"chunk_{i // chunk_duration_3sec + 1}.wav"
            chunk.export(os.path.join(folder_3sec, chunk_filename), format="wav")

        print("Audio splitting into 2-minute and 3-second chunks completed successfully.")
    except Exception as e:
        print(f"Error while splitting audio: {e}")


def extract_mfcc(file_path, duration, sr, offset, n_mfcc):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=duration, sr=sr, offset=offset)

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc), axis=0)
    return mfccs
    
def prepare_features(mfcc_features, required_length):
    current_length = len(mfcc_features)
    if current_length < required_length:
        padded_array = np.pad(mfcc_features, (0, required_length - current_length), 'constant', constant_values=0)
    else:
        padded_array = mfcc_features[:required_length]
    return np.expand_dims(padded_array, axis=0) 

  
def class_predictor():
    folder2 = "webapp\\audiofile-step3-2"
    folder3 = "webapp\\audiofile-step3-3"
    emotion = []
    depression = []
    #I am Loadig file here
    lb_dp = pickle.load(open("lb\\lb-depression.sav", 'rb'))
    lb_emo = pickle.load(open("lb\\lb-emotion.sav", 'rb'))
    model_emotion = load_model("Model\\emotion.keras")
    model_depression = load_model("Model\\depression.keras")


    #Emotion part here
    emo_duration = 3 
    
    for i, file in enumerate(os.listdir(folder3)):
        try:
            file_path = os.path.join(folder3, file)
            mfccs = extract_mfcc(file_path, emo_duration,44100,0.5,13)
            x_testcnn = prepare_features(mfccs,259)

            y_pred = model_emotion.predict(x_testcnn)
            predicted_class = np.argmax(y_pred, axis=1)  
            predicted_emotion = lb_emo.inverse_transform(predicted_class)  
            emotion.append(predicted_emotion[0]) 
        except:
            print(f"{file} is skipped !\n")
      
    print("\n emotion predicted")

    #Depression part here
    dep_duration = 2 * 60  
    for i, file in enumerate(os.listdir(folder2)):
        try:
            file_path = os.path.join(folder2, file)
            mfccs = extract_mfcc(file_path, dep_duration,44100,0.5,20)
            x_testcnn = prepare_features(mfccs ,10293)

            # Predict depression
            y_pred = model_depression.predict(x_testcnn)
            predicted_class = np.argmax(y_pred, axis=1)  
            predicted_depression = lb_dp.inverse_transform(predicted_class)  
            depression.append(predicted_depression[0])
        except:
            print(f"{file} is skipped !\n")
    print("depression completed !")
    TASK_STATUS["completed"] = True 
       

    data={
        "depression":depression,
        "emotion":emotion
    }
    with open("predictions.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    print("*****ML end*****")
   
    
    
def remove_files_in_folder(folder_path):
    
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            
    except Exception as e:
        print(f"An error occurred: {e}")   




def analyze_emotions():
    
    file_path = "predictions.json"
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    
    if "emotion" not in data or not isinstance(data["emotion"], list):
        raise ValueError("The JSON data does not contain a valid 'emotion' list.")
    

    emotion_counts = Counter(data["emotion"])
    # Removing all the files
    output_dir = os.path.join("C:/EL-3rdsem/STRESS/ThirdAttempt/webapp/static", "graphs")
    remove_files_in_folder(output_dir)
    
    
    os.makedirs(output_dir, exist_ok=True)

    # Bar Chart
    bar_chart_path = os.path.join(output_dir, "emotion_bar_chart.png")
    fig = plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.title('Emotion Frequencies', fontsize=16)
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    try:
        fig.savefig(bar_chart_path)
        print(f"Bar chart saved at {bar_chart_path}")
    except Exception as e:
        print(f"Error saving bar chart: {e}")
    plt.close(fig)

    # Line Chart
    line_chart_path=os.path.join(output_dir, "emotion_line_chart.png")
    unique_emotions = ["calm", "fearful", "disgust", "sad", "angry", "happy", "surprised", "neutral"]
    emotion_to_y = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    emotions = data["emotion"]
    time_intervals = range(0, len(emotions) * 3, 3) 
    y_values = [emotion_to_y[emotion] for emotion in emotions]
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_intervals, y_values, marker='o', linestyle='-', color='b', label="Emotion Path")
    plt.yticks(ticks=list(emotion_to_y.values()), labels=unique_emotions)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion")
    plt.title("Emotion Transition Over Time")
    plt.grid(True)
    

    try:
        fig.savefig(line_chart_path)
        print(f"Line chart saved at {line_chart_path}")
    except Exception as e:
        print(f"Error saving line chart: {e}")
    plt.close(fig)
    
    
    #depression 
    dep_line_chart_path=os.path.join(output_dir, "depression_line_chart.png")
    unique_depression = ["low","med","high"]
    depression_to_y = {depression: idx for idx, depression in enumerate(unique_depression)}
    depressions = data["depression"]
    time_intervals = range(0, len(depressions) * 2*60, 2*60)
    y_values = [depression_to_y[depression] for depression in depressions]

    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_intervals, y_values, marker='o', linestyle='-', color='b', label="Depression Path")    
    plt.yticks(ticks=list(depression_to_y.values()), labels=unique_depression)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Depression")
    plt.title("Depreession Transition Over Time")
    plt.grid(True)
    
    try:
        fig.savefig(dep_line_chart_path)
        print(f"Line chart saved at {dep_line_chart_path}")
    except Exception as e:
        print(f"Error saving line chart: {e}")
    plt.close(fig)
    
    
    em_pi_chart_path=os.path.join(output_dir, "em_pi_chart.png")
    fig=plt.figure(figsize=(8, 8))
    plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Emotion Distribution', fontsize=16)

    try:
        fig.savefig(em_pi_chart_path)
        print(f"Line chart saved at {em_pi_chart_path}")
    except Exception as e:
        print(f"Error saving line chart: {e}")
    plt.close(fig)
    
    
    
    
    

    return {
        "bar_chart_path": bar_chart_path,
        "line_chart_path": line_chart_path,
        "counts": dict(emotion_counts),
        "dep_line_chart_path":dep_line_chart_path
        
    }





if __name__ == '__main__':
    app.run(debug=True)
    
