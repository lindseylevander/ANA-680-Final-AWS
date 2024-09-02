import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

def model_fn(model_dir):
    model = RandomForestClassifier(max_depth = 10, random_state= 0, n_estimators = 5)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    return parser.parse_args()

def load_data(train_data_dir):
    data = pd.read_csv(os.path.join(train_data_dir, 'online_gaming_behavior_dataset.csv'))
    X = data.drop(['Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours', 'InGamePurchases', 'GameDifficulty',
                  'EngagementLevel'], axis = 1)
    y = data['EngagementLevel']
    return X, y

def main():
    args = parse_args()
    X_train, y_train = load_data(args.train_data)

    model = model_fn(args.model_dir)

    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, 'RFC_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
