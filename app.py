import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🎯 Customer Churn Prediction Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 3px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .churn-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    .no-churn-risk {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset directly from local directory"""
    try:
        df = pd.read_csv('telco-customer-churn.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'telco-customer-churn.csv' not found in the project directory!")
        st.info("Please make sure the telco-customer-churn.csv file is in your project folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
        
        return model, scaler, feature_names, label_encoders, test_data
        
    except FileNotFoundError as e:
        st.error("❌ Model files not found!")
        st.info("Please run the preprocessing script first: `python scripts/data_preprocessing.py`")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

def create_churn_distribution_chart(df):
    """Create churn distribution visualization"""
    churn_counts = df['Churn'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['No Churn', 'Churn'],
            y=[churn_counts['No'], churn_counts['Yes']],
            marker_color=['#51cf66', '#ff6b6b'],
            text=[f'{churn_counts["No"]} ({churn_counts["No"]/len(df)*100:.1f}%)',
                  f'{churn_counts["Yes"]} ({churn_counts["Yes"]/len(df)*100:.1f}%)'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Customer Churn Distribution",
        xaxis_title="Churn Status",
        yaxis_title="Number of Customers",
        template="plotly_white"
    )
    
    return fig

def create_feature_analysis_charts(df):
    """Create comprehensive feature analysis charts"""
    
    # Numerical features analysis
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    fig_num = make_subplots(
        rows=1, cols=3,
        subplot_titles=numerical_cols,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#667eea', '#764ba2', '#f093fb']
    
    for i, col in enumerate(numerical_cols):
        # Convert TotalCharges to numeric
        if col == 'TotalCharges':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        churn_data = df[df['Churn'] == 'Yes'][col].dropna()
        no_churn_data = df[df['Churn'] == 'No'][col].dropna()
        
        fig_num.add_trace(
            go.Histogram(x=churn_data, name=f'Churn - {col}', 
                        marker_color='rgba(255, 107, 107, 0.7)', 
                        legendgroup='churn', showlegend=(i==0)),
            row=1, col=i+1
        )
        
        fig_num.add_trace(
            go.Histogram(x=no_churn_data, name=f'No Churn - {col}', 
                        marker_color='rgba(81, 207, 102, 0.7)', 
                        legendgroup='no_churn', showlegend=(i==0)),
            row=1, col=i+1
        )
    
    fig_num.update_layout(
        title="Distribution of Numerical Features by Churn Status",
        template="plotly_white",
        height=400,
        barmode='overlay'
    )
    
    return fig_num

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    # Prepare numerical data
    df_num = df.copy()
    df_num['TotalCharges'] = pd.to_numeric(df_num['TotalCharges'], errors='coerce')
    df_num['Churn'] = df_num['Churn'].map({'Yes': 1, 'No': 0})
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr_matrix = df_num[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix of Numerical Features",
        template="plotly_white",
        height=500
    )
    
    return fig

def create_model_performance_charts(test_data):
    """Create model performance visualization"""
    model_results = test_data['model_results']
    
    # Model comparison
    models = list(model_results.keys())
    accuracies = [model_results[model]['accuracy'] for model in models]
    auc_scores = [model_results[model]['auc_score'] for model in models]
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=accuracies,
        marker_color='#667eea',
        text=[f'{acc:.3f}' for acc in accuracies],
        textposition='auto',
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='AUC Score',
        x=models,
        y=auc_scores,
        marker_color='#764ba2',
        text=[f'{auc:.3f}' for auc in auc_scores],
        textposition='auto',
    ))
    
    fig_comparison.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        template="plotly_white",
        barmode='group'
    )
    
    return fig_comparison

def create_roc_curve(test_data):
    """Create ROC curve for the best model"""
    best_model_name = test_data['best_model_name']
    y_test = test_data['y_test']
    y_pred_proba = test_data['model_results'][best_model_name]['y_pred_proba']
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"ROC Curve - {best_model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
    )
    
    return fig

def preprocess_input(input_data, label_encoders, feature_names, scaler):
    """Preprocess user input for prediction"""
    # Create a dataframe with the input
    df_input = pd.DataFrame([input_data])
    
    # Apply label encoding for binary columns
    for col, encoder in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = encoder.transform(df_input[col])
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    
    # Ensure all feature columns are present
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded[feature_names]
    
    # Scale the features
    X_scaled = scaler.transform(df_encoded)
    
    return X_scaled

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Analytics</h1>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    
    if df is None:
        st.stop()
    
    model, scaler, feature_names, label_encoders, test_data = load_model_artifacts()
    
    if model is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Data Overview", "🔍 Exploratory Analysis", "🤖 Model Performance", "🎯 Churn Prediction", "💡 Business Insights"]
    )
    
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>Churn Rate</h3>
                <h2>{churn_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_monthly = pd.to_numeric(df['MonthlyCharges'], errors='coerce').mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Monthly Charges</h3>
                <h2>${avg_monthly:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_tenure = pd.to_numeric(df['tenure'], errors='coerce').mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Tenure (months)</h3>
                <h2>{avg_tenure:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Churn distribution
        st.plotly_chart(create_churn_distribution_chart(df), use_container_width=True)
        
        # Dataset sample
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset info
        st.subheader("ℹ️ Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Missing Values:**", df.isnull().sum().sum())
        
        with col2:
            st.write("**Numerical Columns:**", len(df.select_dtypes(include=[np.number]).columns))
            st.write("**Categorical Columns:**", len(df.select_dtypes(include=['object']).columns))
    
    elif page == "🔍 Exploratory Analysis":
        st.header("🔍 Exploratory Data Analysis")
        
        # Feature distributions
        st.subheader("📈 Feature Distributions by Churn Status")
        st.plotly_chart(create_feature_analysis_charts(df), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
        
        # Categorical feature analysis
        st.subheader("📊 Categorical Features Analysis")
        
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
        
        for col in categorical_cols:
            if col in df.columns:
                fig = px.histogram(
                    df, x=col, color='Churn',
                    title=f'Churn Distribution by {col}',
                    template='plotly_white',
                    color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance Analysis")
        
        if test_data:
            # Model comparison
            st.subheader("📊 Model Comparison")
            st.plotly_chart(create_model_performance_charts(test_data), use_container_width=True)
            
            # ROC Curve
            st.subheader("📈 ROC Curve Analysis")
            st.plotly_chart(create_roc_curve(test_data), use_container_width=True)
            
            # Detailed metrics
            st.subheader("📋 Detailed Performance Metrics")
            
            best_model_name = test_data['best_model_name']
            model_results = test_data['model_results']
            
            col1, col2, col3 = st.columns(3)
            
            for i, (model_name, results) in enumerate(model_results.items()):
                col = [col1, col2, col3][i % 3]
                
                with col:
                    is_best = model_name == best_model_name
                    border_color = "#667eea" if is_best else "#e9ecef"
                    
                    st.markdown(f"""
                    <div style="border: 2px solid {border_color}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <h4>{'🏆 ' if is_best else ''}{model_name}</h4>
                        <p><strong>Accuracy:</strong> {results['accuracy']:.4f}</p>
                        <p><strong>AUC Score:</strong> {results['auc_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    template='plotly_white'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Churn Prediction":
        st.header("🎯 Customer Churn Prediction")
        
        st.markdown("""
        <div class="insight-box">
            <h4>🔮 Predict Customer Churn</h4>
            <p>Enter customer information below to predict the likelihood of churn. Our advanced machine learning model 
            analyzes multiple factors to provide accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("👤 Customer Demographics")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Has Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
            
            with col2:
                st.subheader("📞 Services")
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
            with col3:
                st.subheader("💰 Billing Information")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
            
            submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = {
                    'gender': gender,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': str(total_charges)
                }
                
                try:
                    # Preprocess and predict
                    X_processed = preprocess_input(input_data, label_encoders, feature_names, scaler)
                    prediction = model.predict(X_processed)[0]
                    prediction_proba = model.predict_proba(X_processed)[0]
                    
                    churn_probability = prediction_proba[1] * 100
                    
                    # Display results
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-result churn-risk">
                            ⚠️ HIGH CHURN RISK<br>
                            Probability: {churn_probability:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="insight-box">
                            <h4>🚨 Recommended Actions:</h4>
                            <ul>
                                <li>🎁 Offer retention incentives or discounts</li>
                                <li>📞 Schedule a customer service call</li>
                                <li>📧 Send personalized retention emails</li>
                                <li>🔄 Consider contract upgrade offers</li>
                                <li>💬 Gather feedback on service satisfaction</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-result no-churn-risk">
                            ✅ LOW CHURN RISK<br>
                            Probability: {churn_probability:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="insight-box">
                            <h4> Customer Status: Stable</h4>
                            <ul>
                                <li>🌟 Consider upselling additional services</li>
                                <li>💎 Invite to loyalty program</li>
                                <li>📈 Monitor for service expansion opportunities</li>
                                <li>🗣️ Request referrals and testimonials</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Churn Probability", f"{churn_probability:.1f}%")
                    with col2:
                        st.metric("Retention Probability", f"{100-churn_probability:.1f}%")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    elif page == "💡 Business Insights":
        st.header("💡 Business Insights & Recommendations")
        
        # Key insights
        st.markdown("""
        <div class="insight-box">
            <h3>🎯 Key Findings from Churn Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            ### 📊 Customer Behavior Patterns
            
            **High-Risk Segments:**
            - 📅 Month-to-month contract customers
            - 💳 Electronic check payment users
            - 🌐 Fiber optic internet subscribers
            - 👥 Customers without partners/dependents
            - ⏰ Short tenure customers (< 12 months)
            
            **Retention Factors:**
            - 📋 Long-term contracts (1-2 years)
            - 🏦 Automatic payment methods
            - 👨‍👩‍👧‍👦 Customers with family dependencies
            - 🛡️ Multiple service subscriptions
            """)
        
        with insights_col2:
            st.markdown("""
            ### 💰 Revenue Impact Analysis
            
            **Cost of Churn:**
            - 💸 Average customer lifetime value: $1,500+
            - 📉 Monthly revenue loss per churned customer
            - 🎯 Acquisition cost vs retention cost ratio: 5:1
            
            **Retention ROI:**
            - 📈 5% retention improvement = 25-95% profit increase
            - 🎁 Targeted offers can reduce churn by 15-20%
            - 📞 Proactive outreach increases retention by 30%
            """)
        
        # Strategic recommendations
        st.markdown("""
        <div class="insight-box">
            <h3>🚀 Strategic Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown("""
            ### 🎯 Immediate Actions
            
            1. **Early Warning System**
               - Deploy real-time churn prediction
               - Set up automated alerts for high-risk customers
               - Create intervention workflows
            
            2. **Retention Campaigns**
               - Target month-to-month customers
               - Offer contract upgrade incentives
               - Implement loyalty programs
            """)
        
        with rec_col2:
            st.markdown("""
            ### 📈 Medium-term Strategy
            
            1. **Service Optimization**
               - Improve fiber optic service quality
               - Enhance customer support
               - Streamline billing processes
            
            2. **Customer Experience**
               - Personalized service recommendations
               - Proactive customer communication
               - Feedback collection and action
            """)
        
        with rec_col3:
            st.markdown("""
            ### 🔮 Long-term Vision
            
            1. **Predictive Analytics**
               - Advanced ML model deployment
               - Real-time customer scoring
               - Automated retention actions
            
            2. **Business Intelligence**
               - Customer segmentation strategies
               - Lifetime value optimization
               - Market expansion planning
            """)
        
        # ROI Calculator
        st.markdown("""
        <div class="insight-box">
            <h3>💰 Retention ROI Calculator</h3>
        </div>
        """, unsafe_allow_html=True)
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            customers_at_risk = st.number_input("Customers at Risk", 100, 10000, 1000)
            avg_monthly_revenue = st.number_input("Avg Monthly Revenue per Customer ($)", 20, 200, 65)
        
        with calc_col2:
            retention_rate_improvement = st.slider("Retention Rate Improvement (%)", 5, 50, 15)
            campaign_cost_per_customer = st.number_input("Campaign Cost per Customer ($)", 5, 100, 25)
        
        with calc_col3:
            customers_retained = int(customers_at_risk * retention_rate_improvement / 100)
            total_campaign_cost = customers_at_risk * campaign_cost_per_customer
            annual_revenue_saved = customers_retained * avg_monthly_revenue * 12
            roi = ((annual_revenue_saved - total_campaign_cost) / total_campaign_cost) * 100
            
            st.metric("Customers Retained", f"{customers_retained:,}")
            st.metric("Annual Revenue Saved", f"${annual_revenue_saved:,.0f}")
            st.metric("Campaign ROI", f"{roi:.0f}%")

if __name__ == "__main__":
    main()
