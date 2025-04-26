# app.py

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('login.html')

# Form submission route
@app.route('/submit', methods=['POST'])
def submit():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']

    # Here you can save it to your database (future steps)
    print(f"Received: Username={username}, Password={password}, Email={email}")
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
