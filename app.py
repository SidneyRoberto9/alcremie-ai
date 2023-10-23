from __future__ import annotations

import deepdanbooru as dd
import huggingface_hub
import numpy as np
import PIL.Image
import tensorflow as tf

from typing import Annotated
from fastapi import FastAPI, File, UploadFile


def load_model() -> tf.keras.Model:
    path = huggingface_hub.hf_hub_download(
        'public-data/DeepDanbooru',
        'model-resnet_custom_v3.h5'
    )

    return tf.keras.models.load_model(path)


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(
        'public-data/DeepDanbooru',
        'tags.txt'
    )

    with open(path) as f:
        labels = [line.strip() for line in f.readlines()]

    return labels


model = load_model()
labels = load_labels()
app = FastAPI()


def predict(
    image: PIL.Image.Image,
    score_threshold: float
) -> dict[str, float]:
    _, height, width, _ = model.input_shape
    image = np.asarray(image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True
    )
    image = image.numpy()
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.
    probs = model.predict(image[None, ...])[0]
    probs = probs.astype(float)
    res = dict()

    for prob, label in zip(probs.tolist(), labels):
        if prob < score_threshold:
            continue
        res[label] = prob

    return res


@app.get("/")
async def root():
    return {"message": "Application Has Been Running!!"}


@app.post("/upload/")
async def create_upload_files(trashold: float, picture: Annotated[list[UploadFile], File(...)]):
    lista = []
    for file in picture:
        img = PIL.Image.open(file.file)
        lista.append(list(predict(img, trashold).keys()))
    return lista
