"""
Advanced Analytics Module for Sales Dashboard
Contains Customer Lifetime Value, Churn Prediction, and RFM Analysis functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-learn components
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def sanitize_for_streamlit(df):
    """Convert all object columns to string for Streamlit Arrow compatibility."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'O':
            df_copy[col] = df_copy[col].apply(lambda x: str(x) if not (isinstance(x, (int, float, bool, type(None), pd.Timestamp))) else x)
            if df_copy[col].dtype == 'O':
                df_copy[col] = df_copy[col].astype(str)
    return df_copy

def find_customer_id_column(df):
    """Automatically detect the customer identifier column."""
    # First check for standard customer ID columns
    possible_customer_cols = ['Customer_ID', 'customer_id', 'CustomerID', 'customerId', 'Customer', 'customer', 'Client_ID', 'client_id', 'ClientID', 'User_ID', 'user_id', 'UserID']
    
    for col in possible_customer_cols:
        if col in df.columns:
            return col
    
    # Check for actual columns in the sales data structure
    if 'CustomerName' in df.columns:
        return 'CustomerName'
    elif 'Telephone' in df.columns:
        return 'Telephone'
    
    # If no standard customer column found, look for columns with 'customer', 'client', or 'user' in the name
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['customer', 'client', 'user', 'name', 'phone', 'telephone']):
            return col
    
    return None

def calculate_clv(df):
    """Calculate Customer Lifetime Value for each customer."""
    try:
        # Find customer identifier column
        customer_col = find_customer_id_column(df)
        if customer_col is None:
            st.error("No customer identifier column found. Please ensure your data has a column like 'Customer_ID', 'customer_id', etc.")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Check for required columns
        required_cols = ['net_sale', 'ReceivedAt']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns for CLV calculation: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        st.info(f"Using '{customer_col}' as customer identifier column.")
        
        # Convert ReceivedAt to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['ReceivedAt']):
            df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], errors='coerce')
        
        # Calculate customer metrics
        customer_metrics = df.groupby(customer_col).agg({
            'net_sale': ['sum', 'mean', 'count'],
            'ReceivedAt': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [customer_col, 'total_revenue', 'avg_order_value', 'purchase_frequency', 'first_purchase', 'last_purchase']
        
        # Calculate customer lifespan in days
        customer_metrics['lifespan_days'] = (customer_metrics['last_purchase'] - customer_metrics['first_purchase']).dt.days
        customer_metrics['lifespan_days'] = customer_metrics['lifespan_days'].fillna(0)
        
        # Calculate purchase frequency per year
        customer_metrics['purchase_frequency_yearly'] = customer_metrics['purchase_frequency'] / (customer_metrics['lifespan_days'] / 365 + 1)
        
        # Calculate CLV (simplified formula: AOV * Purchase Frequency * Lifespan in years)
        customer_metrics['clv'] = customer_metrics['avg_order_value'] * customer_metrics['purchase_frequency_yearly'] * (customer_metrics['lifespan_days'] / 365 + 1)
        
        # Segment customers based on CLV
        customer_metrics['clv_segment'] = pd.cut(customer_metrics['clv'], 
                                               bins=3, 
                                               labels=['Low Value', 'Medium Value', 'High Value'])
        
        return customer_metrics
    
    except Exception as e:
        st.error(f"Error calculating CLV: {str(e)}")
        return None

def display_clv_analysis(df):
    """Display Customer Lifetime Value analysis."""
    st.header("💰 Customer Lifetime Value Analysis")
    
    if df is None or df.empty:
        st.warning("No data available for CLV analysis.")
        return
    
    # Calculate CLV
    clv_data = calculate_clv(df)
    
    if clv_data is None:
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_clv = clv_data['clv'].mean()
        st.metric("Average CLV", f"${avg_clv:.2f}")
    
    with col2:
        high_value_customers = len(clv_data[clv_data['clv_segment'] == 'High Value'])
        st.metric("High Value Customers", high_value_customers)
    
    with col3:
        avg_aov = clv_data['avg_order_value'].mean()
        st.metric("Average Order Value", f"${avg_aov:.2f}")
    
    with col4:
        avg_frequency = clv_data['purchase_frequency'].mean()
        st.metric("Avg Purchase Frequency", f"{avg_frequency:.1f}")
    
    # CLV Distribution
    st.subheader("CLV Distribution")
    fig_hist = px.histogram(clv_data, x='clv', nbins=30, title="Customer Lifetime Value Distribution")
    fig_hist.update_layout(xaxis_title="CLV ($)", yaxis_title="Number of Customers")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # CLV by Segment
    st.subheader("CLV by Customer Segment")
    segment_summary = clv_data.groupby('clv_segment').agg({
        'clv': ['mean', 'count'],
        'total_revenue': 'sum'
    }).round(2)
    segment_summary.columns = ['Avg CLV', 'Customer Count', 'Total Revenue']
    st.dataframe(sanitize_for_streamlit(segment_summary.reset_index()), use_container_width=True)
    
    # Top customers by CLV
    st.subheader("Top 20 Customers by CLV")
    top_customers = clv_data.nlargest(20, 'clv')[['Customer_ID', 'clv', 'total_revenue', 'purchase_frequency', 'avg_order_value']]
    st.dataframe(sanitize_for_streamlit(top_customers), use_container_width=True)
    
    # Insights and Recommendations
    st.subheader("💡 CLV Insights & Recommendations")
    
    high_value_pct = (high_value_customers / len(clv_data)) * 100
    st.write(f"• **{high_value_pct:.1f}%** of your customers are high-value customers")
    
    if high_value_pct < 20:
        st.write("• 🎯 **Focus on customer retention**: Implement loyalty programs for high-value customers")
    
    low_clv_customers = len(clv_data[clv_data['clv_segment'] == 'Low Value'])
    if low_clv_customers > len(clv_data) * 0.5:
        st.write("• 📈 **Upselling opportunity**: Many customers have low CLV - consider upselling strategies")
    
    st.write("• 🔄 **Regular monitoring**: Track CLV trends monthly to identify changes in customer behavior")

def predict_churn(df):
    """Predict customer churn using machine learning."""
    if not SKLEARN_AVAILABLE:
        st.error("Scikit-learn is required for churn prediction. Please install it: pip install scikit-learn")
        return None
    
    try:
        # Find customer identifier column
        customer_col = find_customer_id_column(df)
        if customer_col is None:
            st.error("No customer identifier column found. Please ensure your data has a column like 'Customer_ID', 'customer_id', etc.")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Check for required columns
        required_cols = ['net_sale', 'ReceivedAt']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns for churn prediction: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        st.info(f"Using '{customer_col}' as customer identifier column.")
        
        # Convert ReceivedAt to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ReceivedAt']):
            df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], errors='coerce')
        
        # Calculate customer features
        current_date = df['ReceivedAt'].max()
        customer_features = df.groupby(customer_col).agg({
            'net_sale': ['sum', 'mean', 'count'],
            'ReceivedAt': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [customer_col, 'total_spent', 'avg_order_value', 'purchase_count', 'first_purchase', 'last_purchase']
        
        # Calculate recency (days since last purchase)
        customer_features['recency'] = (current_date - customer_features['last_purchase']).dt.days
        
        # Calculate customer lifetime (days between first and last purchase)
        customer_features['lifetime'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
        customer_features['lifetime'] = customer_features['lifetime'].fillna(0)
        
        # Define churn (customers who haven't purchased in the last 90 days)
        customer_features['is_churned'] = (customer_features['recency'] > 90).astype(int)
        
        # Prepare features for ML model
        feature_cols = ['total_spent', 'avg_order_value', 'purchase_count', 'recency', 'lifetime']
        X = customer_features[feature_cols].fillna(0)
        y = customer_features['is_churned']
        
        # Check if we have enough data
        if len(X) < 10:
            st.warning("Not enough customer data for reliable churn prediction (minimum 10 customers required).")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predict churn probability for all customers
        X_scaled = scaler.transform(X)
        churn_proba = rf_model.predict_proba(X)[:, 1]  # Probability of churn
        
        # Add predictions to customer data
        customer_features['churn_probability'] = churn_proba
        customer_features['churn_risk'] = pd.cut(churn_proba, 
                                               bins=[0, 0.3, 0.7, 1.0], 
                                               labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return customer_features, feature_importance, rf_model
    
    except Exception as e:
        st.error(f"Error in churn prediction: {str(e)}")
        return None

def display_churn_prediction(df):
    """Display churn prediction analysis."""
    st.header("🚨 Customer Churn Prediction")
    
    if df is None or df.empty:
        st.warning("No data available for churn prediction.")
        return
    
    # Predict churn
    churn_results = predict_churn(df)
    
    if churn_results is None:
        return
    
    customer_features, feature_importance, model = churn_results
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_count = len(customer_features[customer_features['churn_risk'] == 'High Risk'])
        st.metric("High Risk Customers", high_risk_count)
    
    with col2:
        avg_churn_prob = customer_features['churn_probability'].mean()
        st.metric("Avg Churn Probability", f"{avg_churn_prob:.1%}")
    
    with col3:
        churned_customers = len(customer_features[customer_features['is_churned'] == 1])
        st.metric("Already Churned", churned_customers)
    
    with col4:
        total_customers = len(customer_features)
        st.metric("Total Customers", total_customers)
    
    # Churn risk distribution
    st.subheader("Churn Risk Distribution")
    risk_counts = customer_features['churn_risk'].value_counts()
    fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, title="Customer Churn Risk Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Feature importance
    st.subheader("Churn Prediction - Feature Importance")
    fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                           title="What Factors Predict Customer Churn?")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # High-risk customers
    st.subheader("High-Risk Customers (Immediate Attention Required)")
    high_risk_customers = customer_features[customer_features['churn_risk'] == 'High Risk'].sort_values('churn_probability', ascending=False)
    if not high_risk_customers.empty:
        display_cols = ['Customer_ID', 'churn_probability', 'total_spent', 'recency', 'purchase_count']
        st.dataframe(sanitize_for_streamlit(high_risk_customers[display_cols].head(20)), use_container_width=True)
    else:
        st.info("No high-risk customers identified.")
    
    # Insights and Recommendations
    st.subheader("💡 Churn Prevention Recommendations")
    
    high_risk_pct = (high_risk_count / total_customers) * 100
    st.write(f"• **{high_risk_pct:.1f}%** of customers are at high risk of churning")
    
    if high_risk_count > 0:
        st.write("• 🎯 **Immediate action needed**: Contact high-risk customers with retention offers")
        st.write("• 💌 **Personalized outreach**: Send targeted emails or offers to at-risk customers")
    
    # Top feature insights
    top_feature = feature_importance.iloc[0]['feature']
    st.write(f"• 📊 **Key predictor**: {top_feature} is the strongest predictor of churn")
    
    if 'recency' in feature_importance['feature'].values:
        st.write("• ⏰ **Recency matters**: Days since last purchase is a key churn indicator")

def calculate_rfm(df):
    """Calculate RFM (Recency, Frequency, Monetary) analysis."""
    try:
        # Find customer identifier column
        customer_col = find_customer_id_column(df)
        if customer_col is None:
            st.error("No customer identifier column found. Please ensure your data has a column like 'Customer_ID', 'customer_id', etc.")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Check for required columns
        required_cols = ['net_sale', 'ReceivedAt']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns for RFM analysis: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        st.info(f"Using '{customer_col}' as customer identifier column.")
        
        # Convert ReceivedAt to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ReceivedAt']):
            df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], errors='coerce')
        
        # Calculate RFM metrics
        current_date = df['ReceivedAt'].max()
        
        rfm = df.groupby(customer_col).agg({
            'ReceivedAt': lambda x: (current_date - x.max()).days,  # Recency
            customer_col: 'count',  # Frequency
            'net_sale': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_col, 'Recency', 'Frequency', 'Monetary']
        
        # Create RFM scores (1-5 scale)
        rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5,4,3,2,1])  # Lower recency = higher score
        rfm['F_Score'] = pd.cut(rfm['Frequency'].rank(method='first'), bins=5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.cut(rfm['Monetary'].rank(method='first'), bins=5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate RFM Score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Segment customers
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
                return 'New Customers'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115']:
                return 'Cannot Lose Them'
            elif row['RFM_Score'] in ['331', '321', '231', '241', '251']:
                return 'Hibernating'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customers, axis=1)
        
        return rfm
    
    except Exception as e:
        st.error(f"Error calculating RFM: {str(e)}")
        return None

def display_rfm_analysis(df):
    """Display RFM analysis."""
    st.header("📊 RFM Customer Segmentation")
    
    if df is None or df.empty:
        st.warning("No data available for RFM analysis.")
        return
    
    # Calculate RFM
    rfm_data = calculate_rfm(df)
    
    if rfm_data is None:
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_recency = rfm_data['Recency'].mean()
        st.metric("Avg Recency (days)", f"{avg_recency:.0f}")
    
    with col2:
        avg_frequency = rfm_data['Frequency'].mean()
        st.metric("Avg Frequency", f"{avg_frequency:.1f}")
    
    with col3:
        avg_monetary = rfm_data['Monetary'].mean()
        st.metric("Avg Monetary Value", f"${avg_monetary:.2f}")
    
    with col4:
        total_segments = rfm_data['Segment'].nunique()
        st.metric("Customer Segments", total_segments)
    
    # RFM Segment Distribution
    st.subheader("Customer Segment Distribution")
    segment_counts = rfm_data['Segment'].value_counts()
    fig_segments = px.bar(x=segment_counts.index, y=segment_counts.values, 
                         title="Number of Customers by Segment")
    fig_segments.update_layout(xaxis_title="Customer Segment", yaxis_title="Number of Customers")
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # RFM 3D Scatter Plot
    st.subheader("RFM 3D Visualization")
    fig_3d = px.scatter_3d(rfm_data, x='Recency', y='Frequency', z='Monetary', 
                          color='Segment', title="Customer Segments in 3D RFM Space",
                          hover_data=['Customer_ID'])
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Segment Summary
    st.subheader("Segment Performance Summary")
    segment_summary = rfm_data.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean', 
        'Monetary': ['mean', 'sum'],
        'Customer_ID': 'count'
    }).round(2)
    segment_summary.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Total Revenue', 'Customer Count']
    st.dataframe(sanitize_for_streamlit(segment_summary.reset_index()), use_container_width=True)
    
    # Top customers by segment
    st.subheader("Sample Customers by Segment")
    selected_segment = st.selectbox("Select a segment to view customers:", rfm_data['Segment'].unique())
    segment_customers = rfm_data[rfm_data['Segment'] == selected_segment].head(10)
    display_cols = ['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']
    st.dataframe(sanitize_for_streamlit(segment_customers[display_cols]), use_container_width=True)
    
    # Insights and Recommendations
    st.subheader("💡 RFM Insights & Recommendations")
    
    # Champions
    champions = len(rfm_data[rfm_data['Segment'] == 'Champions'])
    if champions > 0:
        st.write(f"• 🏆 **Champions ({champions} customers)**: Your best customers! Reward them and ask for referrals.")
    
    # At Risk
    at_risk = len(rfm_data[rfm_data['Segment'] == 'At Risk'])
    if at_risk > 0:
        st.write(f"• ⚠️ **At Risk ({at_risk} customers)**: Send personalized campaigns to reconnect.")
    
    # New Customers
    new_customers = len(rfm_data[rfm_data['Segment'] == 'New Customers'])
    if new_customers > 0:
        st.write(f"• 🆕 **New Customers ({new_customers} customers)**: Provide onboarding support and early-stage offers.")
    
    # Hibernating
    hibernating = len(rfm_data[rfm_data['Segment'] == 'Hibernating'])
    if hibernating > 0:
        st.write(f"• 😴 **Hibernating ({hibernating} customers)**: Win them back with special offers or surveys.")
    
    st.write("• 📈 **Regular monitoring**: Update RFM analysis monthly to track customer movement between segments.")
