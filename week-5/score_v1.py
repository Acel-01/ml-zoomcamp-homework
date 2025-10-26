import pickle

# The client data we want to score
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Open the pickled model file in 'read-binary' mode
with open('pipeline_v1.bin', 'rb') as f_in:
    # Load the pipeline from the file
    pipeline = pickle.load(f_in)

# Make the prediction
# We use [client] because the model expects a list of records
prediction = pipeline.predict_proba([client])

# Get the probability of conversion (class 1)
probability_convert = prediction[0, 1]

print(f"The probability that this lead will convert is: {probability_convert:.5f}")