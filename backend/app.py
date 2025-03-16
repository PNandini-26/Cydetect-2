from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import logging
import mysql.connector
from mysql.connector import Error
import os

# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, '..', 'templates'), static_folder=os.path.join(ROOT_DIR, '..', 'static'))
app.secret_key = 'your_secret_key'

# Load the trained models, TF-IDF vectorizers, and label encoders
sentiment_model = joblib.load(os.path.join(ROOT_DIR,  'hack_model.pkl'))
sentiment_tfidf = joblib.load(os.path.join(ROOT_DIR,  'hack_tfidf_vectorizer.pkl'))
cyberbullying_model = joblib.load(os.path.join(ROOT_DIR,  'hack2_model.pkl'))
cyberbullying_tfidf = joblib.load(os.path.join(ROOT_DIR,  'hack2_tfidf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(ROOT_DIR,  'label_encoder.pkl'))

# MySQL Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'nand',
    'database': 'cydetect'
}

# Initialize MySQL database
def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL UNIQUE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                action VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/prediction')
def prediction():
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))
    return render_template('prediction.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'dataset_file' not in request.files:
            return jsonify({'error': 'No file part in the request'})
        
        file = request.files['dataset_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            try:
                # Save the file to a temporary location
                temp_path = os.path.join(ROOT_DIR, 'temp_dataset.csv')
                file.save(temp_path)
                
                # Load the dataset
                new_df = pd.read_csv(temp_path)
                
                # Check if the dataset is for sentiment analysis or cyberbullying detection
                if 'reviews.text' in new_df.columns and 'reviews.rating' in new_df.columns:
                    # Sentiment Analysis using hack_model
                    new_df = new_df.dropna(subset=['reviews.text', 'reviews.rating'])
                    new_df['sentiment'] = new_df['reviews.rating'].apply(lambda x: 'positive' if x > 3 else 'negative')
                    new_X = new_df['reviews.text']
                    new_X_tfidf = sentiment_tfidf.transform(new_X)
                    new_predictions = sentiment_model.predict(new_X_tfidf)
                    prediction_counts = pd.Series(new_predictions).value_counts().to_dict()
                    
                    # Create a circular bar chart
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
                    wedges, texts, autotexts = ax.pie(prediction_counts.values(), wedgeprops=dict(width=0.5), startangle=-40, autopct='%1.1f%%', textprops=dict(color="w"))
                    
                    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
                    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"))
                    for i, p in enumerate(wedges):
                        ang = (p.theta2 - p.theta1)/2. + p.theta1
                        y = np.sin(np.deg2rad(ang))
                        x = np.cos(np.deg2rad(ang))
                        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                        connectionstyle = f"angle,angleA=0,angleB={ang}"
                        kw["arrowprops"].update({"connectionstyle": connectionstyle})
                        ax.annotate(list(prediction_counts.keys())[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                                    horizontalalignment=horizontalalignment, bbox=bbox_props, **kw)
                    
                    ax.set_title("Sentiment Analysis Results")
                    
                    # Save the plot to a BytesIO object
                    img = io.BytesIO()
                    plt.savefig(img, format='png')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode()
                    plt.close(fig)
                    
                    return jsonify({'prediction_counts': prediction_counts, 'plot_url': plot_url})
                elif 'tweet_text' in new_df.columns:
                    # Cyberbullying Detection using hack2_model
                    new_df = new_df.dropna(subset=['tweet_text'])
                    new_X = new_df['tweet_text']
                    new_X_tfidf = cyberbullying_tfidf.transform(new_X)
                    new_predictions = cyberbullying_model.predict(new_X_tfidf)
                    new_predictions_decoded = label_encoder.inverse_transform(new_predictions)
                    prediction_counts = pd.Series(new_predictions_decoded).value_counts().to_dict()
                    
                    return jsonify({'prediction_counts': prediction_counts})
                else:
                    return jsonify({'error': 'Dataset is missing required columns'})
                
                # Remove the temporary file
                os.remove(temp_path)
                
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                return jsonify({'error': str(e)})
    return render_template('analysis.html')


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT id, password FROM users WHERE username = %s', (username,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user and user['password'] == password:
                session['user_id'] = user['id']
                flash('Login successful!')
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password')
                return redirect(url_for('login'))
        except Error as e:
            flash(f'Error: {e}')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password, email) VALUES (%s, %s, %s)', (username, password, email))
            conn.commit()
            cursor.close()
            conn.close()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except Error as e:
            flash(str(e))
            return render_template('register.html')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT action, timestamp FROM history WHERE user_id = %s', (session['user_id'],))
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return render_template('histroy.html', history=history)
    except Error as e:
        flash(str(e))
        return render_template('histroy.html')



if __name__ == '__main__':
    app.run(debug=True)