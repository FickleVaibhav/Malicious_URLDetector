# Importing necessary libraries and modules
import warnings  # Suppress warnings for cleaner output
import joblib  # To load the saved model
import features_extraction  # Module for extracting features from the input URL
import sys  # To read command line arguments
import numpy as np  # For numerical operations

from features_extraction import MODEL_PATH  # Path to the saved model file

# Function to predict the classification of a URL
def predict(test_url):
    # Extract features from the test URL
    features_test = features_extraction.main(test_url)
    features_test = np.array(features_test).reshape((1, -1))  # Reshape to match model input format

    # Load the pre-trained model
    clf = joblib.load(MODEL_PATH)

    # Make prediction
    pred = clf.predict(features_test)
    print(test_url)  # Print the tested URL
    print(pred)  # Print the prediction result
    return int(pred[0])  # Return the prediction result as an integer

# Main function to handle user input and display prediction results
def main():
    url = sys.argv[1]  # Read URL from command line arguments
    # url = "google.com"  # Example URL for testing

    # Make prediction using the predict function
    prediction = predict(url)
    if prediction == 1:
        print("SAFE")  # URL is classified as safe
    else:
        print("PHISHING")  # URL is classified as phishing

# Entry point of the script
if __name__ == "__main__":
    main()  # Execute the main function
