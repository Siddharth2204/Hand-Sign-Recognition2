from flask import Flask, render_template, jsonify
import subprocess
import os

app = Flask(__name__)

# Route to serve the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve the services page (services.html)
@app.route('/services')
def services():
    return render_template('services.html')  # Ensure you have services.html in your templates folder

# Route to trigger the Python script
@app.route('/run-python', methods=['GET'])
def run_python_script():
    script_path = os.path.join(os.getcwd(), "main.py")  # Get the absolute path of the script

    # Check if the script exists
    if os.path.isfile(script_path):
        try:
            # Run the Python script in a separate process
            subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return jsonify({"message": "Opening!"}), 200
        except Exception as e:
            # Return error response if the script fails to start
            return jsonify({"error": f"Failed to start the script: {str(e)}"}), 500
    else:
        # Return error response if the script is not found
        return jsonify({"error": "Python script not found!"}), 404

if __name__ == '__main__':
    app.run(debug=True)
