from crossval import crossval
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_model(training_set, config):
    """Train the machine learning model."""
    models = {}

    # Iterate over sections in the config file
    for section in config.sections():
        # Check if the section corresponds to a supported algorithm
        if section == 'LogisticRegression':
            params = dict(config[section])
            models[section] = LogisticRegression(penalty=params['penalty'], 
                                                 C=float(params['c']), 
                                                 max_iter=int(params['max_iter']))
        elif section == 'RandomForestClassifier':
            params = dict(config[section])
            models[section] = RandomForestClassifier(n_estimators=int(params['n_estimators']), 
                                                     criterion=params['criterion'], 
                                                     max_depth=int(params['max_depth']))
        elif section == 'XGBClassifier':
            params = dict(config[section])
            models[section] = XGBClassifier(n_estimators=int(params['n_estimators']), 
                                            learning_rate=float(params['learning_rate']), 
                                            max_depth=int(params['max_depth']), 
                                            eval_metric=params['eval_metric'])
        else:
            raise ValueError(f"Invalid algorithm: {section}")
    
    X_train = training_set.drop('isFraud', axis=1)  # all columns except 'isFraud'
    y_train = training_set['isFraud']

    # Fit each model separately
    for algo_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)

        # Perform kfold cross validation
        crossval(model_instance, X_train, y_train)

    return models

