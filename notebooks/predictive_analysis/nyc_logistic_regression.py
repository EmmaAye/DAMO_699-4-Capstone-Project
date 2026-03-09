import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, auc

# Define paths and setup output directory
output_dir = "../../output/graphs"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
# Converting to Pandas is standard for sklearn baseline workflows
pdf = spark.table("workspace.capstone_project.nyc_model_ready").toPandas()

# Remove rows with missing label to avoid ValueError
pdf = pdf[pdf['delay_indicator'].notnull()]

# 2. Define Features
numeric_features = ['hour', 'day_of_week', 'month', 'year', 'unified_alarm_level', 'calls_past_30min', 'calls_past_60min']
categorical_features = ['season', 'incident_category', 'unified_call_source', 'location_area']

# 3. Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 4. Train/Test Split
X = pdf.drop(columns=['incident_id', 'incident_datetime', 'response_minutes', 'delay_indicator', 'event_indicator'])
y = pdf['delay_indicator']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train
baseline_model.fit(X_train, y_train)

# 6. Generate and Save Plot
y_probs = baseline_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve - Logistic Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(f"{output_dir}/logistic_roc_curve.png")
print(f"Graph saved to {output_dir}/logistic_roc_curve.png")