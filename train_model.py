from pymongo import MongoClient
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
from config import MONGO_URI, MONGO_DB_NAME

interactioncollection = "interactions"
recommendation_collection = "recommendations"


def fetch_interaction_data():

    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[interactioncollection]

        # Fetch user_id, item_id, score from MongoDB
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
    """
    Fallback: Return top-N most popular items for unknown users.
    Popularity based on average score.
    """
    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(interactions_data, columns=['user_id', 'item_id', 'score'])
    
    # Group by item_id and compute average score
    item_scores = df.groupby('item_id')['score'].mean().sort_values(ascending=False)
    
    # Return top-N items
    return item_scores.head(top_n).index.tolist()

def train_svd_model(interactions_data):
    """
    Train SVD model on interaction data.
    Returns the trained model, trainset, testset, and all items.
    """
    # Define reader for normalized scores (0-5 scale)
    reader = Reader(rating_scale=(0, 5))

    # Convert interactions to DataFrame
    df = pd.DataFrame(interactions_data, columns=['user_id', 'item_id', 'score'])

    # Load data into Surprise
    dataset = Dataset.load_from_df(df, reader)

    # Split into train and test sets (80% train, 20% test)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Initialize SVD model (tuned for small dataset)
    algo = SVD(
        n_factors=20,    # Few factors to avoid overfitting
        n_epochs=20,     # Number of iterations
        lr_all=0.005,    # Learning rate
        reg_all=0.1,     # Regularization to prevent overfitting
        random_state=42
    )

    # Train the model
    algo.fit(trainset)

    # Evaluate on test set
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"Test RMSE: {rmse}")

    # Get all unique item IDs
    all_items = set(df['item_id'].unique())

    return algo, trainset, testset, all_items

# def generate_recommendations(algo, trainset, user_id, all_items, top_n=10, interactions_data=None):

#     try:
#         client = MongoClient(MONGO_URI)
#         db = client[MONGO_DB_NAME]
#         collection = db[recommendation_collection]


       
#         if user_id in trainset._raw2inner_id_users:
#     # Get items the user has already interacted with (use raw item IDs)
#             user_items = set(trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)])

#             # Predict scores for all items not interacted with
#             predictions = []
#             for item_id in all_items:
#                 if item_id not in user_items:
#                     pred = algo.predict(user_id, item_id)
#                     predictions.append((item_id, pred.est))

#             # Sort by predicted score and get top-N
#             predictions.sort(key=lambda x: x[1], reverse=True)
#             top_predictions = predictions[:top_n]
       
#         else:
#             # Fallback for unknown users: recommend popular items
#             print(f"User {user_id} not found in training data. Recommending popular items.")
#             popular_items = get_popular_items(interactions_data, top_n)
#             recommended_items = [
#                 {"item_id": item_id, "score": None}
#                 for item_id in popular_items
#             ]
#             top_predictions = [(item_id, None) for item_id in popular_items]

#         # Store recommendations in MongoDB
#         recommendation_doc = {
#             "user_id": user_id,
#             "recommended_items": recommended_items
#         }
        
#         # Update or insert the document (upsert to avoid duplicates)
#         collection.update_one(
#             {"user_id": user_id},
#             {"$set": recommendation_doc},
#             upsert=True
#         )
        
#         client.close()
#         return top_predictions

#     except Exception as e:
#         print(f"Error storing recommendations in MongoDB: {e}")
#         return top_predictions




def generate_recommendations(algo, trainset, user_id, all_items, top_n=10, interactions_data=None):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[recommendation_collection]  # Fixed: use correct collection name

        # Check if user_id exists in the training set
        if user_id in trainset._raw2inner_id_users:
            # Get items the user has already interacted with (use raw item IDs)
            user_items = set(trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)])

            # Predict scores for all items not interacted with
            predictions = []
            for item_id in all_items:
                if item_id not in user_items:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))

            # Sort by predicted score and get top-N
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:top_n]

            # Define recommended_items for known users
            if top_predictions:
                recommended_items = [
                    {"item_id": item_id, "score": score}
                    for item_id, score in top_predictions
                ]
            else:
                # Fallback if no unrated items (rare)
                print(f"No unrated items for user {user_id}. Recommending popular items.")
                popular_items = get_popular_items(interactions_data, top_n)
                recommended_items = [
                    {"item_id": item_id, "score": None}
                    for item_id in popular_items
                ]
                top_predictions = [(item_id, None) for item_id in popular_items]
        else:
            # Fallback for unknown users: recommend popular items
            print(f"User {user_id} not found in training data. Recommending popular items.")
            popular_items = get_popular_items(interactions_data, top_n)
            recommended_items = [
                {"item_id": item_id, "score": None}
                for item_id in popular_items
            ]
            top_predictions = [(item_id, None) for item_id in popular_items]

        # Store recommendations in MongoDB
        recommendation_doc = {
            "user_id": user_id,
            "recommended_items": recommended_items
        }
        
        # Update or insert the document (upsert to avoid duplicates)
        collection.update_one(
            {"user_id": user_id},
            {"$set": recommendation_doc},
            upsert=True
        )
        
        client.close()
        return top_predictions

    except Exception as e:
        print(f"Error storing recommendations in MongoDB: {e}")
        return top_predictions







def main():
    interactions_data = fetch_interaction_data()

    """Generate recommendations for all users."""
    if not interactions_data:
        print("No interaction data found.")
        return
    algo, trainset, testset, all_items = train_svd_model(interactions_data)
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





if __name__ == "__main__":
    main()