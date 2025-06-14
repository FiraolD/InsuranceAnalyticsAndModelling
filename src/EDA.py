import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Firaol Dabi/OneDrive/Desktop/Insurance-Risk-Analytics/Insurance-Risk-Analytics/Data/cleaned_insurance_data.csv", parse_dates=["TransactionMonth"])  # Update filename
df.rename(columns={"TransactionMonth": "ClaimDate"}, inplace=True)

# Optional: Drop rows with critical missing values for cleaner visuals
df = df.dropna(subset=["ClaimDate", "Province", "VehicleType", "Gender", "make", "Model", "TotalPremium", "TotalClaims", "LossRatio", "CustomValueEstimate"])

plt.figure(figsize=(16, 5))
sns.barplot(data=df, x="Province", y="LossRatio", estimator='mean', errorbar=None)
plt.title("Average Loss Ratio by Province")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="VehicleType", y="LossRatio", estimator='mean', errorbar=None)
plt.title("Average Loss Ratio by Vehicle Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Gender", y="LossRatio", estimator='mean', errorbar=None)
plt.title("Average Loss Ratio by Gender")
plt.tight_layout()
plt.show()

# Claim severity = TotalClaims per incident or per policy ID
claim_severity = df.groupby("make")["TotalClaims"].mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 5))
sns.barplot(x=claim_severity.index, y=claim_severity.values)
plt.title("Average Claim Severity by Vehicle Make (Top 15)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="TotalPremium", y="TotalClaims", hue="VehicleType", alpha=0.7)
plt.title("Premium vs Claims")
plt.xscale("log")  # Optional: log scale if range is wide
plt.yscale("log")
plt.tight_layout()
plt.show()

financial_vars = ["TotalPremium", "TotalClaims", "LossRatio", "CustomValueEstimate"]

plt.figure(figsize=(12, 8))
for i, var in enumerate(financial_vars):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[var], kde=True)
    plt.title(f"Distribution of {var}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(data=df[["TotalClaims", "CustomValueEstimate"]])
plt.title("Outliers in TotalClaims and CustomValueEstimate")
plt.tight_layout()
plt.show()

df['month'] = df['ClaimDate'].dt.to_period("M").astype(str)

monthly = df.groupby("month").agg({
    "TotalClaims": "sum",
    "LossRatio": "mean",
    "PolicyID": "nunique"
}).rename(columns={"PolicyID": "UniquePolicies"})

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly, x=monthly.index, y="TotalClaims", label="Total Claims")
sns.lineplot(data=monthly, x=monthly.index, y="LossRatio", label="Avg Loss Ratio")
plt.xticks(rotation=45)
plt.title("Monthly Claim Frequency and Severity")
plt.tight_layout()
plt.show()

top_makes = df.groupby("make")["TotalClaims"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 5))
top_makes.head(10).plot(kind='bar', color='tomato')
plt.title("Top 10 Vehicle Makes with Highest Total Claims")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
top_makes.tail(10).plot(kind='bar', color='seagreen')
plt.title("Bottom 10 Vehicle Makes with Lowest Total Claims")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
