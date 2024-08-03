# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from detect_ai import predict
from humanize import create_instructions, fix_issues

# Initialize FastAPI app
app = FastAPI()


# Define a data model for the input
class TextInput(BaseModel):
    text: str


# Prediction endpoint
@app.post("/detect_ai")
async def predict_endpoint(input: TextInput):
    text = input.text
    ai_probability = predict(text)
    return {"ai_probability": ai_probability}


# Humanize endpoint
@app.post("/humanize")
async def humanize_endpoint(input: TextInput):
    ai_text = input.text
    instructions = create_instructions(ai_text)
    humanized_text = fix_issues(instructions, ai_text)
    return {"humanized_text": humanized_text}

# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)