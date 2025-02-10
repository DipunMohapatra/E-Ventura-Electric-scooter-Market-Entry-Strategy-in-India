# %%
import pandas as pd
file_path = r"D:\Projects\EV Market Analysis\Competition Analysis.xlsx"
data = pd.read_excel(file_path)
data.info()

# %%
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

# Normalisation using Min-Max Scaling
scaler = MinMaxScaler()
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
scaled_data = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

# Apply weights to columns
weights = {
    'Brand': 0.036,
    'Vehicle Rating': 0.153,
    'Price': 0.189,
    'Range (km/charge)': 0.171,
    'Charging Time (hours)': 0.099,
    'Battery Life (Distance)': 0.108,
    'Cost Per Km': 0.135,
    'Top Speed': 0.081,
    'Aesthetics': 0.036,
    'Service': 0.081,
    'Durability': 0.099
}


scaled_data.columns = scaled_data.columns.str.strip()  # Strip whitespace from column names
for column, weight in weights.items():
    if column in scaled_data.columns:
        scaled_data[column] *= weight

# Calculate ideal best
ideal_best = [
    scaled_data['Brand'].max(),
    scaled_data['Vehicle Rating'].max(),
    scaled_data['Price'].min(),
    scaled_data['Range (km/charge)'].max(),
    scaled_data['Charging Time (hours)'].min(),
    scaled_data['Cost Per Km'].min(),
    scaled_data['Top Speed'].max(),
    scaled_data['Aesthetics'].max(),
    scaled_data['Service'].max(),
    scaled_data['Durability'].max(),
]

# Calculate ideal worst
ideal_worst = [
    scaled_data['Brand'].min(),
    scaled_data['Vehicle Rating'].min(),
    scaled_data['Price'].max(),
    scaled_data['Range (km/charge)'].min(),
    scaled_data['Charging Time (hours)'].max(),
    scaled_data['Cost Per Km'].max(),
    scaled_data['Top Speed'].min(),
    scaled_data['Aesthetics'].min(),
    scaled_data['Service'].min(),
    scaled_data['Durability'].min(),
]

# Convert ideal best and worst to NumPy arrays
ideal_best = np.array(ideal_best).reshape(1, -1)
ideal_worst = np.array(ideal_worst).reshape(1, -1)

# Compute distances
best_distance = cdist(scaled_data.values, ideal_best, metric='euclidean').flatten()
worst_distance = cdist(scaled_data.values, ideal_worst, metric='euclidean').flatten()

#Evaluate performance scores: worst_distance/best_distance + worst_distance
performance_score = worst_distance/(best_distance + worst_distance)
# Add distances to the DataFrame
scaled_data['Distance to Best'] = best_distance
scaled_data['Distance to Worst'] = worst_distance
scaled_data['Performance Scores'] = performance_score
scaled_data['Model'] = data['Model']

scaled_data.to_excel("D:\Topsis.xlsx", index=False)




