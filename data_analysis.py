import pandas as pd

# Path to your attributes.csv
csv_path = "data/celeba-subset/train/attributes.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Show the first few rows and columns
print("Columns:", df.columns.tolist())
print(df.head())

# Count positive/negative for each attribute
attr_counts = df.iloc[:, 1:].apply(pd.Series.value_counts).T
attr_counts.columns = ['Negative', 'Positive']
print("\nAttribute counts (Negative/Positive):")
print(attr_counts)

# Show proportion of positives for each attribute
attr_props = (df.iloc[:, 1:] == 1).mean().sort_values(ascending=False)
print("\nProportion of images with each attribute (descending):")
print(attr_props)

# Optionally: show combinations of top attributes
top_attrs = attr_props.head(5).index.tolist()
print(f"\nSample counts for top 5 attributes: {top_attrs}")
print(df[top_attrs].value_counts().head(10))

# Optionally: plot histogram
import matplotlib.pyplot as plt
attr_props.plot(kind='bar', figsize=(12,4), title='Proportion of images with each attribute')
plt.tight_layout()
plt.show()