import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1️⃣ Load dataset
# =========================
df = pd.read_csv("Fashion_Retail_Sales.csv")
print("✅ Dataset loaded successfully\n")

# =========================
# 2️⃣ Data Cleaning
# =========================
# Fill missing values
df['Purchase Amount (USD)'] = df['Purchase Amount (USD)'].fillna(df['Purchase Amount (USD)'].mean())
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].median())

# Clean categorical columns
df['Item Purchased'] = df['Item Purchased'].str.strip().str.title()
df['Payment Method'] = df['Payment Method'].str.strip().str.title()

# Convert Date Purchase to datetime and remove invalid dates
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'], errors='coerce')
df = df.dropna(subset=['Date Purchase'])

# Remove duplicates
df = df.drop_duplicates()

# Set Date Purchase as index for time series
df.set_index('Date Purchase', inplace=True)

print("✅ Data Cleaning done\n")
print("Final dataset info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head(), "\n")

# =========================
# 3️⃣ Descriptive Statistics
# =========================
print("✅ Descriptive Statistics\n")

# Mean
print("Mean Purchase Amount:", df['Purchase Amount (USD)'].mean())
print("Mean Review Rating:", df['Review Rating'].mean(), "\n")

# Median
print("Median Purchase Amount:", df['Purchase Amount (USD)'].median())
print("Median Review Rating:", df['Review Rating'].median(), "\n")

# Mode
print("Mode of Purchase Amount:", df['Purchase Amount (USD)'].mode()[0])
print("Mode of Review Rating:", df['Review Rating'].mode()[0], "\n")

# Standard Deviation
print("Standard Deviation of Purchase Amount:", df['Purchase Amount (USD)'].std())
print("Standard Deviation of Review Rating:", df['Review Rating'].std(), "\n")

# Min & Max
print("Minimum Purchase Amount:", df['Purchase Amount (USD)'].min())
print("Maximum Purchase Amount:", df['Purchase Amount (USD)'].max())
print("Minimum Review Rating:", df['Review Rating'].min())
print("Maximum Review Rating:", df['Review Rating'].max(), "\n")

# Total rows
print("Total rows in dataset:", df.shape[0], "\n")

# =========================
# 4️⃣ Time Series Analysis
# =========================
# Monthly Sales
monthly_sales = df['Purchase Amount (USD)'].resample('ME').sum()
print("Monthly Sales:\n", monthly_sales, "\n")

# Yearly Sales
yearly_sales = df['Purchase Amount (USD)'].resample('YE').sum()
print("Yearly Sales:\n", yearly_sales, "\n")

# Plot Monthly Sales
plt.figure(figsize=(12,6))
monthly_sales.plot(marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Purchase Amount (USD)')
plt.grid(True)
plt.show()

# Plot Yearly Sales
plt.figure(figsize=(8,5))
yearly_sales.plot(marker='o', color='orange')
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Purchase Amount (USD)')
plt.grid(True)
plt.show()

# 3-Month Rolling Average
monthly_sales_rolling = monthly_sales.rolling(window=3).mean()
plt.figure(figsize=(12,6))
monthly_sales_rolling.plot()
plt.title('3-Month Rolling Average of Sales')
plt.xlabel('Month')
plt.ylabel('Purchase Amount (USD)')
plt.grid(True)
plt.show()

# =========================
# 5️⃣ Customer & Product Analysis
# =========================
# Customer Analysis
print("✅ Customer Analysis\n")
total_customers = df['Customer Reference ID'].nunique()
print("Total Unique Customers:", total_customers)

purchase_freq = df.groupby('Customer Reference ID')['Purchase Amount (USD)'].count()
print("\nPurchase frequency per customer:\n", purchase_freq.head())

avg_purchase = df.groupby('Customer Reference ID')['Purchase Amount (USD)'].mean()
print("\nAverage purchase per customer:\n", avg_purchase.head())

top_customers = df.groupby('Customer Reference ID')['Purchase Amount (USD)'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Customers by Total Spending:\n", top_customers)

# Plot Top Customers
plt.figure(figsize=(10,5))
top_customers.plot(kind='bar', color='green', title='Top 10 Customers by Spending')
plt.xlabel('Customer ID')
plt.ylabel('Total Purchase Amount (USD)')
plt.show()

# Product Analysis
print("✅ Product Analysis\n")
top_products_count = df['Item Purchased'].value_counts().head(10)
print("\nTop 10 Products by Purchase Count:\n", top_products_count)

top_products_revenue = df.groupby('Item Purchased')['Purchase Amount (USD)'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Revenue:\n", top_products_revenue)

# Plot Top Products by Count
plt.figure(figsize=(10,5))
top_products_count.plot(kind='bar', color='skyblue', title='Top 10 Products by Purchase Count')
plt.xlabel('Product')
plt.ylabel('Number of Purchases')
plt.show()

# Plot Top Products by Revenue
plt.figure(figsize=(10,5))
top_products_revenue.plot(kind='bar', color='orange', title='Top 10 Products by Revenue')
plt.xlabel('Product')
plt.ylabel('Total Revenue (USD)')
plt.show()

# Payment Method Analysis
payment_method_counts = df['Payment Method'].value_counts()
print("\nPayment Method Distribution:\n", payment_method_counts)

# Plot Payment Methods
plt.figure(figsize=(6,6))
payment_method_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue','lightgreen','orange','pink'])
plt.title('Payment Method Distribution')
plt.ylabel('')
plt.show()

# =========================
# 6️⃣ Recommendations
# =========================
print("✅ Recommendations based on EDA\n")
print("1. Focus marketing and loyalty programs on top 10 customers to boost repeat purchases.")
print("2. Promote top-selling products (e.g., Handbag, Dress, Tunic) as they generate maximum revenue.")
print("3. Encourage popular payment methods (like Credit Card) since most customers prefer them.")
print("4. Plan promotions during low-sales months identified in the monthly sales trend.")
print("5. Improve product reviews for high-selling items with lower ratings to enhance customer satisfaction.")

# =========================
# 10️⃣ Save Cleaned Data
# =========================
clean_file = "Fashion_Retail_Sales_Cleaned.csv"
df.to_csv(clean_file)
print(f"✅ Cleaned dataset saved successfully as '{clean_file}'")
