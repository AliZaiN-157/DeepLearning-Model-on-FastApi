from PIL import Image
from prediction import predict
import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()


@app.get('/')
async def index():
    return {"Hello": "World Api"}


@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    Catagories = ['Fire', 'Smoke']
    result = predict(image)
    prediction = Catagories[int(result[0][0])]
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
