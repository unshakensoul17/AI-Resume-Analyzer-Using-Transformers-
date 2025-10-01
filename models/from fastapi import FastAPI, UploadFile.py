from fastapi import FastAPI, UploadFile
import torch
import pickle

# Import your custom classes
from best_model import ColabResumeScorer, ColabResumeAnalyzer

app = FastAPI()

# Load model state dict and scaler once at startup
model = ColabResumeScorer()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize analyzer with loaded model and scaler or paths
analyzer = ColabResumeAnalyzer(model=model, scaler=scaler)


@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile, job_description: str = ""):
    file_bytes = await file.read()
    result = analyzer.analyze_resume_bytes(file_bytes, file.filename, job_description)
    return result
