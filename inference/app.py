from fastapi import FastAPI, status, Request
from inference.model.model import ModelInput, ModelOutput

app = FastAPI()

@app.post("/inference", status_code=status.HTTP_201_CREATED, response_model=ModelOutput)
def inference(request: Request):
    tweet: ModelInput = request.body()
    print(tweet)

