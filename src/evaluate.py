from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def evaluate_model(model, test_set):
      X_test = test_set.drop('isFraud', axis=1)  # all columns except 'isFraud'
      y_test = test_set['isFraud']

      """Evaluate the model's performance."""

      for algo_name, model_instance in model.items():
            y_pred = model_instance.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')  # Specify averaging strategy
            recall = recall_score(y_test, y_pred, average='weighted')  # Specify averaging strategy
            cm = confusion_matrix(y_test, y_pred)  # Confusion matrix for multiclass problems

            print(f"\nPerformance Evaluation for {algo_name} model."
                  f"\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")    
            print("\nTraining Confusion Matrix"
                  f"\n{cm}")
            print("-" * 40)