from flask import Flask, request, render_template
import joblib
import numpy as np
import features_extraction
from features_extraction import MODEL_PATH

# Initialize Flask app
app = Flask(__name__ , template_folder=r"C:\Users\fickl\Desktop\final_model\mal_link\test\templates")

# Prediction function
def predict(test_url):
    try:
        # Extract features using your existing module
        features_test = features_extraction.main(test_url)
        features_test = np.array(features_test).reshape((1, -1))

        # Load the pre-trained model
        clf = joblib.load(MODEL_PATH)

        # Predict the result
        pred = clf.predict(features_test)
        return "SAFE" if int(pred[0]) == 1 else "PHISHING/MALICIOUS"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # Placeholder for result
    if request.method == "POST":
        # Get URL from the form
        url = request.form.get("url")
        if url:
            # Call the prediction function
            result = predict(url)
        else:
            result = "Please provide a valid URL."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
