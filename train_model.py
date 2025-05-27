from pymongo import MongoClient
from surprise import SVD, Dataset, Reader
import pandas as pd
from bson import ObjectId
from config import MONGO_URI, MONGO_DB_NAME

interactioncollection = "interactions"
recommendation_collection = "recommendations"


def fetch_interaction_data():
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[interactioncollection]
        interactions_data = [
            (str(doc["user_id"]), str(doc["item_id"]), float(doc["score"]))
            for doc in collection.find()
        ]
        client.close()
        return interactions_data
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return []


def get_popular_items(interactions_data, top_n=10):
    df = pd.DataFrame(interactions_data, columns=['user_id', 'item_id', 'score'])
    item_scores = df.groupby('item_id')['score'].mean().sort_values(ascending=False)
    return item_scores.head(top_n).index.tolist()


def train_svd_model(interactions_data):
    reader = Reader(rating_scale=(0, 5))
    df = pd.DataFrame(interactions_data, columns=['user_id', 'item_id', 'score'])
    dataset = Dataset.load_from_df(df, reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(
        n_factors=20,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.1,
        random_state=42
    )
    algo.fit(trainset)
    all_items = set(df['item_id'].unique())
    return algo, trainset, all_items


def generate_recommendations(algo, trainset, user_id, all_items, top_n=10, interactions_data=None):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[recommendation_collection]

        if user_id in trainset._raw2inner_id_users:
            user_items = set(trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)])
            predictions = []
            for item_id in all_items:
                if item_id not in user_items:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:top_n]
            if top_predictions:
                recommended_items = [
                    {"item_id": item_id, "score": score}
                    for item_id, score in top_predictions
                ]
            else:
                print(f"No unrated items for user {user_id}. Recommending popular items.")
                popular_items = get_popular_items(interactions_data, top_n)
                recommended_items = [
                    {"item_id": item_id, "score": None}
                    for item_id in popular_items
                ]
                top_predictions = [(item_id, None) for item_id in popular_items]
        else:
            print(f"User {user_id} not found in training data. Recommending popular items.")
            popular_items = get_popular_items(interactions_data, top_n)
            recommended_items = [
                {"item_id": item_id, "score": None}
                for item_id in popular_items
            ]
            top_predictions = [(item_id, None) for item_id in popular_items]

        recommendation_doc = {
            "user_id": user_id,
            "recommended_items": recommended_items
        }

        collection.update_one(
            {"user_id": user_id},
            {"$set": recommendation_doc},
            upsert=True
        )
        client.close()
        return top_predictions

    except Exception as e:
        print(f"Error storing recommendations in MongoDB: {e}")
        return []


def refresh_recommendations():
    interactions_data = fetch_interaction_data()
    if not interactions_data:
        print("No interaction data found.")
        return
    algo, trainset, all_items = train_svd_model(interactions_data)
    df = pd.DataFrame(interactions_data, columns=['user_id', 'item_id', 'score'])
    all_users = df['user_id'].unique()
    print(f"Generating top-10 recommendations for {len(all_users)} users...")
    for user_id in all_users:
        recommendations = generate_recommendations(
            algo, trainset, user_id, all_items, top_n=10, interactions_data=interactions_data
        )
        print(f"Recommendations for user {user_id}:")
        for item_id, score in recommendations:
            if score is not None:
                print(f"  Item {item_id}: Predicted score {score:.2f}")
            else:
                print(f"  Item {item_id}: Popular item (no score)")
    print("All recommendations stored in MongoDB 'recommendations' collection.")



