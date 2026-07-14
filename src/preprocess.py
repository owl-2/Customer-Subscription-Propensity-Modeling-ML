import pandas as pd

def feature_engineering(train_data):

    # Remove ID if present
    if 'ID' in train_data.columns:
        train_data = train_data.drop('ID', axis=1)

    # Age Group
    train_data['age_group'] = pd.cut(
        train_data['age'],
        bins=[18, 30, 40, 50, 60, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '60+']
    )

    # Balance Group
    train_data['balance_group'] = pd.cut(
        train_data['balance'],
        bins=[-float('inf'), 0, 1000, 5000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    # Contact Intensity
    train_data['contact_intensity'] = (
        train_data['campaign'] +
        train_data['previous']
    )

    # Campaign Efficiency
    train_data['campaign_efficiency'] = (
        train_data['duration'] /
        (train_data['campaign'] + 1)
    )

    return train_data