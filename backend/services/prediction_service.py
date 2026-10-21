from collections import Counter

def format_prediction_response(model_results):
    predictions = [res["prediction"] for res in model_results.values()]
    
    # Majority vote
    vote_counts = Counter(predictions)
    majority_class, majority_votes = vote_counts.most_common(1)[0]
    
    total_models = len(model_results)
    agreement = majority_votes / total_models if total_models > 0 else 0.0
    
    is_cancer = (majority_class != "normal")
    
    return {
        "models": model_results,
        "majority_vote": {
            "prediction": majority_class,
            "votes": majority_votes,
            "total_models": total_models,
            "agreement": agreement
        },
        "is_cancer": is_cancer
    }
