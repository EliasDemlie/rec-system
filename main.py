import uvicorn
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timezone
from mangum import Mangum
import os
from train_model import refresh_recommendations
from contextlib import asynccontextmanager

from config import MONGO_URI, MONGO_DB_NAME, PRODUCT_DB_NAME
from recommendation import get_item_details, recommend_items_by_content, recommend_by_last_interacted, vectorize_item

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server started...")
    yield
    print("Server shutting down...")

app = FastAPI(
    title="Recommendation API",
    description="API for recording interactions and generating recommendations",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
product_db = client[PRODUCT_DB_NAME]

INTERACTIONS_COLLECTION = db["interactions"]
RECOMMENDATIONS_COLLECTION = db["recommendations"]
PRODUCT_COLLECTION = product_db["products"]
VECTOR_COLLECTION = db["item_vectors"]

# Interaction weights
EVENT_WEIGHTS = {
    "click": 0.5,
    "add_to_cart": 0.8,
    "purchase": 1.0,
    "rating": None
}

# Models
class Interaction(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    rating: float | None = None

class Item(BaseModel):
    name: str
    description: str
    category: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Recommendation API"}

@app.post("/interactions/record")
async def record_interaction(interaction: Interaction):
    score = EVENT_WEIGHTS.get(interaction.event_type, 0.5)
    if interaction.event_type == "rating" and interaction.rating is not None:
        score = round(interaction.rating * 0.2, 1)

    existing = INTERACTIONS_COLLECTION.find_one({
        "user_id": interaction.user_id,
        "item_id": interaction.item_id
    })

    if existing:
        new_score = min(existing["score"] + score, 5.0)
        INTERACTIONS_COLLECTION.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "score": new_score,
                    "timestamp": datetime.now(timezone.utc)
                }
            }
        )
    else:
        INTERACTIONS_COLLECTION.insert_one({
            "user_id": interaction.user_id,
            "item_id": interaction.item_id,
            "score": min(score, 5.0),
            "timestamp": datetime.now(timezone.utc)
        })

    return {"message": "Interaction recorded"}

@app.get("/user/recommendations/{user_id}")
async def get_stored_recommendations(user_id: str = Path(...)):
    if not user_id or user_id.lower() == "undefined":
        raise HTTPException(status_code=400, detail="Invalid or missing user_id")

    recommendation = RECOMMENDATIONS_COLLECTION.find_one({"user_id": user_id})
    if recommendation:
        sorted_items = sorted(recommendation["recommended_items"], key=lambda x: -x["score"])
        item_ids = [item["item_id"] for item in sorted_items]
        detailed_items = get_item_details(item_ids)
        return {
            "user_id": user_id,
            "recommended_items": detailed_items
        }

    fallback_items = recommend_by_last_interacted(user_id)
    if not fallback_items:
        return {"message": "User has not interacted with any items yet."}

    return {
        "user_id": user_id,
        "recommended_items": fallback_items,
        "note": "Generated from top interaction"
    }

@app.get("/item/similar/{item_id}")
async def get_content_similar_items(item_id: str = Path(...)):
    if not item_id or item_id.lower() == "undefined":
        raise HTTPException(status_code=400, detail="Invalid or missing item_id")

    try:
        recommendations = recommend_items_by_content(item_id)
        return {"recommended_items": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during similarity calculation: {str(e)}"
        )


@app.get("/refresh-recommendations")
async def refresh_recommendations_endpoint():
    try:
        refresh_recommendations()
        return {"message": "Recommendations refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing recommendations: {str(e)}")






handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
