import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timezone
from mangum import Mangum
import os
from contextlib import asynccontextmanager  # Correct import for lifespan

from config import MONGO_URI, MONGO_DB_NAME, PRODUCT_DB_NAME
from recommendation import get_item_details, recommend_items_by_content, recommend_by_last_interacted, vectorize_item

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    print("Server started...")
    yield
    print("Server shutting down...")

# FastAPI app initialization with lifespan
app = FastAPI(
    title="Recommendation API",
    description="API for recording interactions and generating recommendations",
    lifespan=lifespan
)

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
product_db = client[PRODUCT_DB_NAME]

# Collections
INTERACTIONS_COLLECTION = db["interactions"]
RECOMMENDATIONS_COLLECTION = db["recommendations"]
PRODUCT_COLLECTION = product_db["products"]
VECTOR_COLLECTION = db["item_vectors"]

# Event weights for scoring interactions
EVENT_WEIGHTS = {
    "click": 0.5,
    "add_to_cart": 0.8,
    "purchase": 1.0,
    "rating": None
}

# Pydantic models
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
    """Record a user interaction and update or insert it in the database."""
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
async def get_stored_recommendations(user_id: str):
    """Retrieve stored recommendations or generate fallback recommendations for a user."""
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

@app.get("/items/similar/{item_id}")
async def get_content_similar_items(item_id: str):
    """Retrieve content-based similar items for a given item ID."""
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


# Mangum handler for Vercel
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if not set
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)