import io

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List

from starlette.responses import StreamingResponse

from utils import make_inference

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.model_dump()])
    pred = make_inference(data)
    return pred


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    # Чтение файла в формате CSV и его обработка
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    pred = make_inference(data)
    data['selling_price'] = pred

    stream = io.StringIO()
    data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response
