# ============================================
# Introduction to Machine Learning
# Titanic Dataset Exploration & Cleaning
# ============================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Titanic dataset
df = sns.load_dataset("titanic")

# 3. Understand the dataset
print("FIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE (rows, columns):")
print(df.shape)

print("\nDATA INFO:")
print(df.info())

print("\nCOLUMN DESCRIPTIONS:")
print(df.columns)

# 4. Summary statistics
print("\nNUMERICAL SUMMARY:")
print(df.describe())

print("\nCATEGORICAL VALUE COUNTS:")
print("\nSex:\n", df["sex"].value_counts())
print("\nEmbarked:\n", df["embarked"].value_counts())
print("\nClass:\n", df["class"].value_counts())

# Groupby example
print("\nSURVIVAL RATE BY CLASS:")
print(df.groupby("class")["survived"].mean())

print("\nSURVIVAL RATE BY SEX:")
print(df.groupby("sex")["survived"].mean())

# 5. Missing data analysis
print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nMISSING VALUE PERCENTAGE:")
print(df.isnull().mean() * 100)

# Suggested strategies (printed explanation)
print("\nMISSING DATA STRATEGY:")
print("- Age: fill with median")
print("- Embarked: fill with mode")
print("- Deck: drop column (too many missing values)")

# 6. Data Visualization

# Age distribution
plt.figure()
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Sex distribution
plt.figure()
sns.countplot(x="sex", data=df)
plt.title("Gender Distribution")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# Survival distribution
plt.figure()
df["survived"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Survival Distribution")
plt.ylabel("")
plt.show()

# Survival by Pclass
plt.figure()
sns.barplot(x="class", y="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# Survival by Sex
plt.figure()
sns.barplot(x="sex", y="survived", data=df)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

# Survival by Embarked
plt.figure()
sns.barplot(x="embarked", y="survived", data=df)
plt.title("Survival Rate by Embarked Port")
plt.ylabel("Survival Rate")
plt.show()

# 7. Data Cleaning

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Drop deck (too many missing values)
df = df.drop(columns=["deck"])

# Encode categorical variables
df["sex"] = df["sex"].map({"male": 0, "female": 1})

df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

print("\nDATA AFTER CLEANING:")
print(df.info())

# 8. Basic Insights

survival_rate = df["survived"].mean() * 100
print(f"\nSURVIVAL PERCENTAGE: {survival_rate:.2f}%")

print("\nSURVIVAL BY CLASS:")
print(df.groupby("class")["survived"].mean())

print("\nSURVIVAL BY SEX:")
print(df.groupby("sex")["survived"].mean())

# Age vs survival
print("\nAVERAGE AGE BY SURVIVAL:")
print(df.groupby("survived")["age"].mean())

# 9. Bonus: Feature Engineering

df["family_size"] = df["sibsp"] + df["parch"]
df["is_alone"] = (df["family_size"] == 0).astype(int)

print("\nFEATURE ENGINEERING:")
print(df[["family_size", "is_alone"]].head())

# 10. Bonus Visualizations

# Pairplot (numerical features)
sns.pairplot(df[["age", "fare", "family_size", "survived"]])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 11. Mini Predictive Rule
print("\nMINI DECISION RULE:")
print("If (Class == First) AND (Sex == Female) → Higher survival probability")
print("Else → Lower survival probability")

# 12. Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")
