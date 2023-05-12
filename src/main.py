import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from train import TrainModel
from predict import Predict, Product

app = FastAPI()

# create a class for product


class ProductModel(BaseModel):
    brand: str
    current_price: int
    original_price: int
    discount_percentage: float
    rating: int
    num_ratings: int
    model_name: str
    dial_shape: str
    strap_color: str
    strap_material: str
    touchscreen: str
    battery_life_days: str
    bluetooth: str
    display_size: str
    weight: str


@app.get('/')
def home():
    return {'message': 'api is wroking!'}


@app.post('/predict')
def predict(product: ProductModel):
    pr = Product(
        battery_life_days=product.battery_life_days,
        bluetooth=product.bluetooth,
        brand=product.brand,
        current_price=product.current_price,
        dial_shape=product.display_size,
        model_name=product.model_name,
        discount_percentage=product.discount_percentage,
        display_size=product.display_size,
        num_ratings=product.num_ratings,
        original_price=product.original_price,
        rating=product.rating,
        strap_color=product.strap_color,
        strap_material=product.strap_material,
        touchscreen=product.touchscreen,
        weight=product.weight
    )
    p = Predict()
    pr = p.predict_row(pr)
    tm = TrainModel.load_model()
    pred = tm.predict(pr.drop(['Discount Price'], axis=1))
    pred = int(pred[0])
    return {'result': pred}


if __name__ == "__main__":
    uvicorn.run(app)
