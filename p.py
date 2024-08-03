import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

# Find similar patterns
def find_similar_pattern(data, N):
    current_pattern = data[-N:]
    min_distance = float('inf')
    similar_pattern_index = -1

    for i in range(len(data) - 2 * N):
        past_pattern = data[i:i + N]
        distance = euclidean_distances([current_pattern], [past_pattern])[0][0]
        
        if distance < min_distance:
            min_distance = distance
            similar_pattern_index = i

    return similar_pattern_index, min_distance

# Predict future prices
def predict(data, similar_pattern_index, N):
    return data[similar_pattern_index + N:similar_pattern_index + 2 * N]

# Plot the patterns
def plot_patterns(current_pattern, similar_pattern):
    plt.figure(figsize=(12, 6))
    plt.plot(current_pattern, label='Current Pattern')
    plt.plot(similar_pattern, label='Similar Pattern')
    plt.legend()
    plt.show()

# Main function
def main(file_path, N):
    data = load_data(file_path)['Close']
    current_pattern = data[-N:].values
    similar_pattern_index, _ = find_similar_pattern(data.values, N)
    similar_pattern = data[similar_pattern_index:similar_pattern_index + N].values
    predicted_prices = predict(data.values, similar_pattern_index, N)

    print(f"Similar pattern found at index: {similar_pattern_index}")
    print(f"Predicted prices: {predicted_prices}")

    plot_patterns(current_pattern, similar_pattern)

# Example usage
file_path = 'bitcoin_price.csv'  # replace with your actual file path
N = 30  # number of days to consider for the pattern
main(file_path, N)