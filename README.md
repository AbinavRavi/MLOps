# MLOps
MLOps Level 4 project

## Plan

### Step 1 - Training model and torchscript conversion and model registration

**Task at hand**: Sentiment Analysis

**Dataset used**: [Twitter tweets data](https://www.kaggle.com/datasets/kazanova/sentiment140)

**Skills to be learnt**: 
- Sentiment analysis modelling using pytorch lightning
- Early stopping and tests to verify if model metrics are satisfying (for automatic retraining )
- MLFlow for experiment tracking and hyper parameter search 
- MLFlow model registration and versioning


## Future work

- Create an API for inference of text and quantify performance using FastAPI and pydantic models for support
- Dockerize the application
- Add github actions 
- Add streaming of the tweets for faster inference
- Add a simple frontend 
- Add a monitoring layer which monitors the predictions and triggers an alert if something is astray
