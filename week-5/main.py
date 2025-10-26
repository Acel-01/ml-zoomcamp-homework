import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load the model when the server starts
with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# 2. Initialize the FastAPI app
app = FastAPI()

# 3. Define the input data shape using Pydantic
# This ensures the data we receive is valid
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# 4. Define our prediction endpoint
@app.post("/score")
def score_client(client: Client):
    # Convert the Pydantic Client object to a dictionary
    client_dict = client.dict()
    
    # Make the prediction
    prediction = pipeline.predict_proba([client_dict])
    probability_convert = prediction[0, 1]
    
    # Return the result as JSON
    return {"conversion_probability": probability_convert}


# def main():
#     print("Hello from week-5!")
# if __name__ == "__main__":
#     main()
