from sklearn.model_selection import cross_val_score

def crossval(model, X_train, y_train):
    """k-fold cross-validation"""
    # k is amendable #
    k=5
    
    scores = cross_val_score(model, X_train, y_train, cv=k, scoring="accuracy")
    # Printing the k-fold cross-validation scores
    print(f"\n{k}-Fold Cross-Validation Scores for {model}:")
    for i, score in enumerate(scores, 1):
        print(f"Fold-{i}: {score:.3f}")
    print(f"Average Accuracy: {scores.mean():.3f}")
    print(f"Standard Deviation: {scores.std():.3f}")
    
    return