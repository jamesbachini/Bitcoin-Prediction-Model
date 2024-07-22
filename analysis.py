import pandas as pd

# Load the data
data = pd.read_csv('bitcoin_prices.csv')

# Function to print out statistics
def print_stats(df):
    numeric_df = df.select_dtypes(include=[float, int])
    stats = numeric_df.describe().transpose()
    stats['median'] = numeric_df.median()
    stats['std'] = numeric_df.std()
    stats = stats[['mean', 'median', 'std', 'min', 'max']]
    
    for col in stats.index:
        print(f"{col}:")
        print(f"  Mean: {stats.at[col, 'mean']}")
        print(f"  Median: {stats.at[col, 'median']}")
        print(f"  Standard Deviation: {stats.at[col, 'std']}")
        print(f"  Min: {stats.at[col, 'min']}")
        print(f"  Max: {stats.at[col, 'max']}")
        print("")

# Print stats for all numeric columns
print_stats(data)
