import os
import sqlite3
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Function to fetch data from the database
def fetch_locations():
    # Use an absolute or relative path to the SQLite database file
    db_path = 'D:/flask_app/Database/news_data.db'  # Adjust this path as needed
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Delete rows with empty values
    cursor.execute("DELETE FROM news WHERE latitude IS NULL OR longitude IS NULL OR locations IS NULL OR link IS NULL OR latitude = '' OR longitude = '' OR locations = '' OR link = ''")
    conn.commit()  # Commit the delete operation
    
    # Fetch remaining data after deletion
    cursor.execute("SELECT latitude, longitude, locations, link FROM news")
    locations = cursor.fetchall()
    
    conn.close()
    return locations

@app.route('/api/locations')
def locations():
    data = fetch_locations()
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
