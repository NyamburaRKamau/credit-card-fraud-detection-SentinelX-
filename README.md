🧠 AI-Powered Credit Card Fraud Detection System

💡 Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning.
The dataset used is highly imbalanced, with legitimate transactions vastly outnumbering fraudulent ones.
To address this, I used SMOTE (Synthetic Minority Oversampling Technique) to balance the data and built a robust Random Forest model to classify transactions accurately.

⸻

⚙️ Tech Stack
• Languages: Python
• Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn
• Model: RandomForestClassifier
• IDE: VS Code

⸻

📊 Project Workflow 1. Data Loading & Exploration:
Loaded the Kaggle dataset (creditcard.csv) and explored class imbalance. 2. Preprocessing:
• Scaled features using StandardScaler
• Applied SMOTE to balance fraud vs non-fraud cases 3. Model Training:
• Used RandomForestClassifier (n_estimators=100, random_state=42)
• Trained on balanced data for improved minority class recognition 4. Evaluation:
• Measured Accuracy, Precision, Recall, and F1-Score
• Visualized Confusion Matrix and Feature Importance

📈 Results
Metric Score
Accuracy 99.94%
Precision (Fraud) 0.89
Recall (Fraud) 0.79
F1 Score (Fraud) 0.84

✅ Fraud detection recall improved significantly with SMOTE, identifying 79% of actual frauds.
✅ Minimal false positives ensured financial decision reliability.

⸻

🔍 Visual Insights
Include screenshots (after uploading):
• Confusion Matrix Heatmap
• Top Feature Importance Chart

⸻

🧩 Key Learnings
• Handling imbalanced datasets in real-world financial systems.
• Balancing model performance between precision and recall.
• Building a deployable pipeline from raw CSV → trained ML model.

🏁 How to Run

# Navigate to folder

cd fraud-detection-system

# Install dependencies

pip install -r requirements.txt

# Run project

python main.py

🧾 Dataset Source
Kaggle - Credit Card Fraud Detection Dataset

⸻
