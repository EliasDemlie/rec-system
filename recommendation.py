from typing import List
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from typing import List, Dict, Any  # Ensure Dict is imported here


from config import MONGO_URI, MONGO_DB_NAME, PRODUCT_DB_NAME

# Initialize spaCy
vectorizer = spacy.load("en_core_web_md")

# MongoDB connection
client = MongoClient(MONGO_URI)
mongo_db = client[MONGO_DB_NAME]
product_db = client[PRODUCT_DB_NAME]

# Collections
INTERACTION_COLLECTION = mongo_db["interactions"]
VECTOR_COLLECTION = mongo_db["item_vectors"]
PRODUCT_COLLECTION = product_db["products"]
RECOMMENDATION_COLLECTION = mongo_db["user_recommendations"]


class Item(BaseModel):
    id: str
    name: str
    description: str
    category: str


# def get_item_details(item_ids: List[str]) -> List[dict]:
#     """Retrieve item details from product collection for given item IDs."""
#     object_ids = [ObjectId(item_id) for item_id in item_ids]
#     cursor = PRODUCT_COLLECTION.find({"_id": {"$in": object_ids}})
#     items = []
#     for doc in cursor:
#         doc["_id"] = str(doc["_id"])
#         items.append(doc)
#     return items

def get_item_details(item_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieve item details from the product collection for given item IDs, including category details.
    
    Args:
        item_ids (List[str]): List of product IDs to fetch.
        
    Returns:
        List[Dict[str, Any]]: A list of product documents with resolved category details.
    """
    try:
        # Convert item_ids to ObjectIds for MongoDB query
        object_ids = [ObjectId(item_id) for item_id in item_ids]

        # Define the aggregation pipeline
        pipeline = [
            {
                "$match": {
                    "_id": {"$in": object_ids}  # Filter by the provided item IDs
                }
            },
            {
                "$lookup": {
                    "from": "categories",  # The target collection
                    "localField": "category",  # Field in products collection
                    "foreignField": "_id",  # Field in categories collection
                    "as": "category_details"  # Output array field
                }
            },
            {
                "$unwind": "$category_details"  # Deconstruct the category_details array into a single object
            },
            {
                "$project": {
                    "_id": {"$toString": "$_id"},  # Convert _id to string
                    "name": 1,
                    "description": 1,
                    "brand": 1,
                    "images": 1,
                    "inStock": 1,
                    "quantity": 1,
                    "price": 1,
                    "category": {"$toString": "$category"},  # Convert category ObjectId to string
                    "category_name": "$category_details.name",
                    "sub_categories": "$category_details.subCategories",
                    "icon": "$category_details.icon",
                    "created_at": {
                        "$dateToString": {
                            "format": "%Y-%m-%d %H:%M:%S",
                            "date": "$category_details.createdAt"
                        }
                    },
                    "updated_at": {
                        "$dateToString": {
                            "format": "%Y-%m-%d %H:%M:%S",
                            "date": "$category_details.updatedAt"
                        }
                    }
                }
            }
        ]

        # Execute the aggregation pipeline
        cursor = PRODUCT_COLLECTION.aggregate(pipeline)

        # Convert results to a list and handle images
        items = []
        for doc in cursor:
            # Convert _id in each image document to string
            if "images" in doc:
                for image in doc["images"]:
                    if "_id" in image and isinstance(image["_id"], ObjectId):
                        image["_id"] = str(image["_id"])
                    elif "_id" in image and isinstance(image["_id"], dict) and "$oid" in image["_id"]:
                        image["_id"] = image["_id"]["$oid"]
            items.append(doc)

        return items

    except Exception as e:
        print(f"Error fetching item details: {e}")
        return []





def vectorize_item(text: str) -> List[float]:
    """Convert text to vector using spaCy, with preprocessing."""
    doc = vectorizer(text)
    cleaned_tokens = [
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct
    ]
    cleaned_text = " ".join(cleaned_tokens)
    item_vector = vectorizer(cleaned_text).vector
    return item_vector.tolist()


def recommend_items_by_content(item_id: str, top_n: int = 10) -> List[dict]:
    """Generate content-based recommendations for a given item."""
    target_doc = VECTOR_COLLECTION.find_one({"_id": item_id})

    if not target_doc:
        target_item = PRODUCT_COLLECTION.find_one({"_id": ObjectId(item_id)})
        if not target_item:
            raise ValueError(f"Item not found in both collections with id: {item_id}")
        category = target_item.get("category", "")
        description = target_item.get("description", "")
        text = f"{category} {description}".strip()
        item_vector = vectorize_item(text)

        if item_vector is None or not item_vector:
            raise ValueError(f"Vectorization failed for item: {item_id}")
        target_doc = {"_id": item_id, "vector": item_vector}
        VECTOR_COLLECTION.insert_one(target_doc)

    target_vector = np.array(target_doc["vector"]).reshape(1, -1)
    other_docs = VECTOR_COLLECTION.find({"_id": {"$ne": item_id}})
    similarities = []

    for doc in other_docs:
        other_id = doc["_id"]
        other_vector = np.array(doc["vector"]).reshape(1, -1)
        score = cosine_similarity(target_vector, other_vector)[0][0]
        similarities.append((other_id, score))

    top_similar = sorted(similarities, key=lambda x: -x[1])[:top_n]
    item_ids = [str(other_id) for other_id, _ in top_similar]
    print(f"Top similar items for {item_id}: {item_ids}")
    detailed_items = get_item_details(item_ids)
    id_score_map = {str(other_id): float(score) for other_id, score in top_similar}

    for item in detailed_items:
        item["score"] = id_score_map.get(item["_id"], 0.0)
    detailed_items.sort(key=lambda x: -x["score"])
    detailed_items = [
        {k: v for k, v in item.items() if k != "score"} for item in detailed_items
    ]

    return detailed_items


def recommend_by_last_interacted(user_id: str) -> List[dict] | None:
    """Recommend items based on the user's last interaction."""
    top_doc = INTERACTION_COLLECTION.find_one(
        {"user_id": user_id}, sort=[("score", -1)]
    )
    if not top_doc:
        return None

    top_item_id = top_doc["item_id"]
    recommended_items_id = recommend_items_by_content(top_item_id)
    recommended_items = get_item_details(recommended_items_id)
    return recommended_items


