import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import pickle
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    """Load and preprocess the telco customer churn dataset"""
    try:
        # Load data directly from local CSV file
        df = pd.read_csv('telco-customer-churn.csv')
        print(f"📊 Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'telco-customer-churn.csv' file not found!")
        print("Please make sure the dataset file is in your project directory.")
        raise FileNotFoundError("Dataset file not found")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        raise
    
    # Store original data for analysis
    original_df = df.copy()
    
    # Drop unnecessary columns
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"🔧 Missing values before cleaning: {df.isnull().sum().sum()}")
    df.dropna(inplace=True)
    print(f"🔧 Missing values after cleaning: {df.isnull().sum().sum()}")
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Store label encoders for later use
    label_encoders = {}
    
    # Encode binary columns
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    print(f"🏷️ Encoding {len(binary_cols)} binary columns...")
    
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # One-hot encode remaining categorical variables
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    print(f"🏷️ One-hot encoding {len(categorical_cols)} categorical columns...")
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    print(f"📈 Final dataset shape after encoding: {df_encoded.shape}")
    
    # Features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    print(f"🎯 Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
    
    # Feature scaling
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle class imbalance with SMOTE
    print("🔄 Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    print(f"📊 After SMOTE - Shape: {X_res.shape}, Target distribution: {pd.Series(y_res).value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    print(f"✂️ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return {
        'original_df': original_df,
        'processed_df': df_encoded,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'label_encoders': label_encoders
    }

def train_models(data):
    """Train multiple models and return the best one"""
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print("\n🤖 Training multiple models...")
    print("="*50)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    }
    
    model_results = {}
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    # Select best model based on AUC score
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc_score'])
    best_model = model_results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"🎯 Best AUC Score: {model_results[best_model_name]['auc_score']:.4f}")
    
    return best_model, model_results, best_model_name

if __name__ == "__main__":
    print("🚀 Starting Customer Churn Prediction Model Training")
    print("="*60)
    
    try:
        # Load and preprocess data
        print("📥 Loading and preprocessing data...")
        data = load_and_preprocess_data()
        
        # Train models
        print("\n🤖 Training models...")
        best_model, model_results, best_model_name = train_models(data)
        
        # Save artifacts
        print("\n💾 Saving model artifacts...")
        
        # Save best model
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print("✅ Best model saved")
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(data['scaler'], f)
        print("✅ Scaler saved")
        
        # Save feature names
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(data['feature_names'], f)
        print("✅ Feature names saved")
        
        # Save label encoders
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(data['label_encoders'], f)
        print("✅ Label encoders saved")
        
        # Save test data for evaluation
        with open('test_data.pkl', 'wb') as f:
            pickle.dump({
                'X_test': data['X_test'],
                'y_test': data['y_test'],
                'model_results': model_results,
                'best_model_name': best_model_name
            }, f)
        print("✅ Test data saved")
        
        print("\n" + "="*60)
        print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"🏆 Best Model: {best_model_name}")
        print(f"🎯 Accuracy: {model_results[best_model_name]['accuracy']:.4f}")
        print(f"📈 AUC Score: {model_results[best_model_name]['auc_score']:.4f}")
        print("\n🚀 You can now run: streamlit run app.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure 'telco-customer-churn.csv' is in your project directory")
        print("2. Check that all required packages are installed")
        print("3. Ensure you have write permissions in the project directory")
        raise
