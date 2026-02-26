import pandas as pd

# Load dataset
data = pd.read_excel("data/Online Retail.xlsx")

# Show first 5 rows
print(data.head())

# Dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())

# Remove missing CustomerID
data = data.dropna(subset=['CustomerID'])

# Remove cancelled orders (Invoice starting with C)
data = data[~data['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative quantities
data = data[data['Quantity'] > 0]

print("Cleaned Data Shape:", data.shape)

# Create Customer-Product Matrix
customer_product_matrix = data.pivot_table(
    index='CustomerID',
    columns='Description',
    values='Quantity',
    aggfunc='sum'
).fillna(0)

print(customer_product_matrix.head())

from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between products
product_similarity = cosine_similarity(customer_product_matrix.T)

print("Similarity matrix created")

# Create recommendation function
def recommend_products(product_name, num_recommendations=5):
    product_index = customer_product_matrix.columns.get_loc(product_name)
    similarity_scores = list(enumerate(product_similarity[product_index]))

    # Sort products by similarity
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get recommended products
    recommended_products = [
        customer_product_matrix.columns[i[0]]
        for i in similarity_scores[1:num_recommendations+1]
    ]

    return recommended_products


# Example recommendation
product = customer_product_matrix.columns[0]
print("Recommendations for:", product)
print(recommend_products(product))