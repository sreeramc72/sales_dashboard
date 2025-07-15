import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector
from google.cloud import bigquery
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import logging
import os
from dotenv import load_dotenv

# Suppress warnings and configure logging
warnings.filterwarnings('ignore')

# Configure logging to suppress WebSocket errors
logging.getLogger('tornado.access').setLevel(logging.ERROR)
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('tornado.general').setLevel(logging.ERROR)

# Suppress specific WebSocket errors
class WebSocketErrorFilter(logging.Filter):
    def filter(self, record):
        # Filter out WebSocket closed errors
        if 'WebSocketClosedError' in str(record.getMessage()):
            return False
        if 'Stream is closed' in str(record.getMessage()):
            return False
        if 'write_message' in str(record.getMessage()):
            return False
        return True

# Apply filter to relevant loggers
for logger_name in ['tornado.websocket', 'tornado.iostream']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(WebSocketErrorFilter())
    logger.setLevel(logging.ERROR)

# Load environment variables
load_dotenv("C:\\Users\\sreer\\OneDrive\\Desktop\\mysql\\.env")

# Configure page
st.set_page_config(
    page_title="Growth Team Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def add_calculated_columns(df):
    """Add calculated columns for enhanced analysis"""
    if df.empty:
        return df
    
    # ✅ CUSTOM NET_SALE CALCULATED COLUMN
    # Net Sale calculation based on your business logic
    df['net_sale'] = np.where(
        df['GrossPrice'] == df['Discount'],  # If gross sale equals discount
        (df['GrossPrice'] / 1.05) - (df['Discount'] / 1.05),  # Apply formula with VAT adjustment
        df['GrossPrice'] / 1.05  # Otherwise, just gross sale divided by 1.05
    )
    
    # Financial Performance Metrics
    df['Profit_Margin'] = ((df['net_sale'] - df['Discount']) / df['GrossPrice'] * 100).fillna(0)
    df['Discount_Percentage'] = (df['Discount'] / df['GrossPrice'] * 100).fillna(0)
    df['Revenue_After_Delivery'] = df['net_sale'] - df['Delivery']
    df['Total_Fees'] = df['Delivery'] + df['Surcharge'] + df['VAT']
    df['Order_Profitability'] = df['net_sale'] - df['Discount'] - df['Total_Fees']
    
    # Order Categories
    df['Order_Size_Category'] = pd.cut(df['net_sale'], 
                                     bins=[0, 25, 75, 150, float('inf')],
                                     labels=['Small', 'Medium', 'Large', 'XLarge'])
    
    df['Discount_Category'] = pd.cut(df['Discount'], 
                                   bins=[0, 1, 10, 25, float('inf')],
                                   labels=['No Discount', 'Low', 'Medium', 'High'])
    
    # Time-based Analysis
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    df['Time_Period'] = pd.cut(df['Hour'], 
                              bins=[0, 6, 12, 18, 24],
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    # Customer Value Metrics (if CustomerName exists)
    if 'CustomerName' in df.columns:
        customer_stats = df.groupby('CustomerName')['net_sale'].agg(['count', 'sum', 'mean'])
        customer_stats.columns = ['Customer_Order_Count', 'Customer_Total_Spent', 'Customer_Avg_Order']
        df = df.merge(customer_stats, left_on='CustomerName', right_index=True, how='left')
        
        df['Customer_Value_Tier'] = pd.cut(df['Customer_Total_Spent'],
                                         bins=[0, 100, 500, 1000, float('inf')],
                                         labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    # Efficiency Metrics
    df['Revenue_Per_Item'] = df['net_sale'] / df.get('ItemPrice', 1).replace(0, 1)
    df['Tip_Percentage'] = (df['Tips'] / df['net_sale'] * 100).fillna(0)
    
    # Business Performance Indicators
    df['High_Value_Order'] = df['net_sale'] > df['net_sale'].quantile(0.8)
    df['Discounted_Order'] = df['Discount'] > 0
    df['Premium_Customer'] = df.get('Customer_Order_Count', 0) > 5
    
    return df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_mysql(days_back=7):
    """Load complete data from MySQL database with configurable date range"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )
        
        # Complete data loading - no limits for comprehensive analysis
        if days_back is None:
            # Load all data when specifically requested
            query = """
            SELECT * FROM sales_data 
            ORDER BY ReceivedAt DESC
            """
        else:
            # Load complete data for specified number of days
            query = f"""
            SELECT * FROM sales_data 
            WHERE ReceivedAt >= DATE_SUB(NOW(), INTERVAL {days_back} DAY)
            ORDER BY ReceivedAt DESC
            """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Data preprocessing
        df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'])
        df['Date'] = df['ReceivedAt'].dt.date
        df['Hour'] = df['ReceivedAt'].dt.hour
        df['DayOfWeek'] = df['ReceivedAt'].dt.day_name()
        df['Month'] = df['ReceivedAt'].dt.month_name()
        df['Quarter'] = df['ReceivedAt'].dt.quarter
        
        # Clean numeric columns
        numeric_cols = ['ItemPrice', 'Surcharge', 'Delivery', 'net_sale', 'GrossPrice', 'Discount', 'VAT', 'Total', 'Tips']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add calculated columns for enhanced analysis
        df = add_calculated_columns(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes for better performance
def load_data_from_bigquery(days_back=7):
    """Load complete data from BigQuery with configurable date range"""
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            "C:\\Users\\sreer\\OneDrive\\Desktop\\Dont delete\\"
            "my-database-order-level-2025-92bf0e71cddc.json"
        )
        
        client = bigquery.Client()
        
        # Show progress indicator
        with st.spinner('🔄 Loading data from BigQuery...'):
            if days_back is None:
                st.info("📊 Loading all available data...")
                query = """
                SELECT * FROM `my-database-order-level-2025.order_level.sales_data`
                WHERE ReceivedAt IS NOT NULL 
                ORDER BY ReceivedAt DESC
                """
            else:
                st.info(f"📊 Loading COMPLETE data for last {days_back} days...")
                # Note: BigQuery date filtering will be done in Python after timestamp conversion
                # due to different timestamp formats - load recent data and filter in Python
                query = """
                SELECT * FROM `my-database-order-level-2025.order_level.sales_data`
                WHERE ReceivedAt IS NOT NULL 
                ORDER BY ReceivedAt DESC
                """
            
            df = client.query(query).to_dataframe()
        
        if len(df) == 0:
            st.warning("⚠️ No data returned from BigQuery")
            return pd.DataFrame()
        
        st.info(f"📊 Loaded {len(df):,} records from BigQuery, now processing timestamps...")
        
        # Data preprocessing - Handle different timestamp formats with validation
        if 'ReceivedAt' in df.columns:
            try:
                st.info(f"📊 Processing {len(df):,} records with timestamp conversion...")
                
                # Try different timestamp conversion methods based on data type and values
                if df['ReceivedAt'].dtype in ['int64', 'float64']:
                    # Detect timestamp unit by examining the values
                    sample_value = df['ReceivedAt'].iloc[0]
                    st.info(f"🔍 Sample timestamp value: {sample_value}")
                    
                    # If value is very large (> 1e15), likely microseconds
                    if sample_value > 1e15:
                        st.info("🕐 Detected: Microseconds format")
                        df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], unit='us', errors='coerce')
                    # If value is moderately large (> 1e12), likely milliseconds
                    elif sample_value > 1e12:
                        st.info("🕐 Detected: Milliseconds format")
                        df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], unit='ms', errors='coerce')
                    # If value is smaller, likely seconds
                    else:
                        st.info("🕐 Detected: Seconds format")
                        df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], unit='s', errors='coerce')
                else:
                    # If it's already string/datetime, convert normally
                    st.info("🕐 Converting from string/datetime format")
                    df['ReceivedAt'] = pd.to_datetime(df['ReceivedAt'], errors='coerce')
                
                # Remove any rows where timestamp conversion failed
                before_conversion = len(df)
                df = df[df['ReceivedAt'].notna()].copy()
                after_conversion = len(df)
                
                if before_conversion > after_conversion:
                    st.warning(f"⚠️ Removed {before_conversion - after_conversion:,} records with invalid timestamps")
                
                if len(df) == 0:
                    st.error("❌ No valid timestamps found after conversion")
                    return pd.DataFrame()
                
                # Additional validation: ensure timestamps are within reasonable range
                min_date = pd.Timestamp('2020-01-01')
                max_date = pd.Timestamp('2030-12-31')
                before_range_filter = len(df)
                df = df[
                    (df['ReceivedAt'] >= min_date) & 
                    (df['ReceivedAt'] <= max_date)
                ].copy()
                after_range_filter = len(df)
                
                if before_range_filter > after_range_filter:
                    st.info(f"📅 Filtered {before_range_filter - after_range_filter:,} records outside 2020-2030 range")
                
                if len(df) == 0:
                    st.warning("⚠️ No data with valid timestamps in reasonable date range (2020-2030)")
                    return pd.DataFrame()
                
                # Show date range of loaded data
                min_date_loaded = df['ReceivedAt'].min()
                max_date_loaded = df['ReceivedAt'].max()
                st.success(f"✅ Successfully processed timestamps! Date range: {min_date_loaded.strftime('%Y-%m-%d')} to {max_date_loaded.strftime('%Y-%m-%d')}")
                
                # Apply date filtering in Python if days_back is specified
                if days_back is not None:
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                    before_filter = len(df)
                    df = df[df['ReceivedAt'] >= cutoff_date].copy()
                    after_filter = len(df)
                    if before_filter > after_filter:
                        st.success(f"� Complete data filter: {after_filter:,} records from last {days_back} days (filtered out {before_filter - after_filter:,} older records)")
                
            except Exception as e:
                st.error(f"❌ Error converting ReceivedAt timestamps: {e}")
                return pd.DataFrame()
        
        df['Date'] = df['ReceivedAt'].dt.date
        df['Hour'] = df['ReceivedAt'].dt.hour
        df['DayOfWeek'] = df['ReceivedAt'].dt.day_name()
        df['Month'] = df['ReceivedAt'].dt.month_name()
        df['Quarter'] = df['ReceivedAt'].dt.quarter
        
        # Clean numeric columns
        numeric_cols = ['ItemPrice', 'Surcharge', 'Delivery', 'net_sale', 'GrossPrice', 'Discount', 'VAT', 'Total', 'Tips']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add calculated columns for enhanced analysis
        df = add_calculated_columns(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from BigQuery: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate key performance metrics"""
    if df.empty:
        return {}
    
    metrics = {
        'total_orders': len(df),
        'total_revenue': df['net_sale'].sum(),
        'avg_order_value': df['net_sale'].mean(),
        'total_customers': df['CustomerName'].nunique(),
        'total_discount': df['Discount'].sum(),
        'discount_rate': (df['Discount'].sum() / df['GrossPrice'].sum()) * 100 if df['GrossPrice'].sum() > 0 else 0,
        'avg_delivery_fee': df['Delivery'].mean(),
        'total_tips': df['Tips'].sum()
    }
    
    return metrics

def create_sales_trends_chart(df):
    """Create sales trends visualization"""
    if df.empty:
        return go.Figure()
    
    daily_sales = df.groupby('Date').agg({
        'net_sale': 'sum',
        'OrderID': 'count',
        'Discount': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Revenue Trend', 'Daily Orders Count', 'Discount Distribution', 'Revenue vs Orders'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Daily Revenue
    fig.add_trace(
        go.Scatter(x=daily_sales['Date'], y=daily_sales['net_sale'],
                  mode='lines+markers', name='Revenue', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Daily Orders
    fig.add_trace(
        go.Bar(x=daily_sales['Date'], y=daily_sales['OrderID'],
               name='Orders', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    # Discount Distribution
    fig.add_trace(
        go.Histogram(x=df['Discount'], name='Discount Distribution',
                    marker_color='#2ca02c', nbinsx=30),
        row=2, col=1
    )
    
    # Revenue vs Orders scatter
    fig.add_trace(
        go.Scatter(x=daily_sales['OrderID'], y=daily_sales['net_sale'],
                  mode='markers', name='Revenue vs Orders',
                  marker=dict(size=8, color='#d62728')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Sales Trends Analysis")
    return fig

def create_customer_behavior_analysis(df):
    """Analyze customer behavior patterns"""
    if df.empty:
        return go.Figure(), pd.DataFrame()
    
    # Customer analysis
    customer_stats = df.groupby('CustomerName').agg({
        'net_sale': ['sum', 'mean', 'count'],
        'Discount': 'sum',
        'ReceivedAt': ['min', 'max']
    }).round(2)
    
    customer_stats.columns = ['Total_Spent', 'Avg_Order_Value', 'Order_Count', 'Total_Discount', 'First_Order', 'Last_Order']
    customer_stats = customer_stats.reset_index()
    
    # Customer segments
    customer_stats['Customer_Lifetime_Days'] = (customer_stats['Last_Order'] - customer_stats['First_Order']).dt.days
    customer_stats['Segment'] = pd.cut(customer_stats['Total_Spent'], 
                                     bins=[0, 100, 500, 1000, float('inf')],
                                     labels=['Low Value', 'Medium Value', 'High Value', 'VIP'])
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Customer Segments', 'Order Frequency Distribution', 
                       'Customer Lifetime Value', 'Repeat Customer Analysis'),
        specs=[[{"type": "pie"}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Customer segments pie chart
    segment_counts = customer_stats['Segment'].value_counts()
    fig.add_trace(
        go.Pie(labels=segment_counts.index, values=segment_counts.values,
               name="Customer Segments"),
        row=1, col=1
    )
    
    # Order frequency
    fig.add_trace(
        go.Histogram(x=customer_stats['Order_Count'], name='Order Frequency',
                    marker_color='#ff7f0e', nbinsx=20),
        row=1, col=2
    )
    
    # Customer lifetime value
    fig.add_trace(
        go.Scatter(x=customer_stats['Order_Count'], y=customer_stats['Total_Spent'],
                  mode='markers', name='CLV Analysis',
                  marker=dict(size=8, color='#2ca02c')),
        row=2, col=1
    )
    
    # Repeat customers
    repeat_customers = customer_stats[customer_stats['Order_Count'] > 1]
    repeat_rate = len(repeat_customers) / len(customer_stats) * 100
    
    fig.add_trace(
        go.Bar(x=['New Customers', 'Repeat Customers'],
               y=[len(customer_stats) - len(repeat_customers), len(repeat_customers)],
               name='Customer Type', marker_color=['#d62728', '#9467bd']),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Customer Behavior Analysis")
    
    return fig, customer_stats

def create_discount_performance_analysis(df):
    """Analyze discount performance and effectiveness"""
    if df.empty:
        return go.Figure()
    
    # Discount analysis
    discount_df = df[df['Discount'] > 0].copy()
    
    if discount_df.empty:
        st.warning("No discount data available for analysis")
        return go.Figure()
    
    # Group by discount ranges
    discount_df['Discount_Range'] = pd.cut(discount_df['Discount'], 
                                         bins=[0, 10, 25, 50, 100, float('inf')],
                                         labels=['1-10', '11-25', '26-50', '51-100', '100+'])
    
    discount_analysis = discount_df.groupby('Discount_Range').agg({
        'net_sale': ['sum', 'mean'],
        'OrderID': 'count',
        'Discount': 'mean'
    }).round(2)
    
    # Flatten column names for clarity
    discount_analysis.columns = ['Total_Net_Sales', 'Average_Order_Value', 'Number_of_Orders', 'Average_Discount_Amount']
    discount_analysis = discount_analysis.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Discount Distribution by Amount', 'Revenue by Discount Range',
                       'Discount Code Performance', 'Discount Impact on Order Value'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Discount distribution
    fig.add_trace(
        go.Histogram(x=discount_df['Discount'], name='Discount Distribution',
                    marker_color='#1f77b4', nbinsx=30),
        row=1, col=1
    )
    
    # Revenue by discount range
    discount_revenue = discount_df.groupby('Discount_Range')['net_sale'].sum()
    fig.add_trace(
        go.Bar(x=discount_revenue.index, y=discount_revenue.values,
               name='Revenue by Discount Range', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    # Discount code performance
    if 'DiscountCode' in discount_df.columns:
        top_codes = discount_df.groupby('DiscountCode')['net_sale'].sum().nlargest(10)
        fig.add_trace(
            go.Bar(x=top_codes.index, y=top_codes.values,
                   name='Top Discount Codes', marker_color='#2ca02c'),
            row=2, col=1
        )
    
    # Discount impact
    fig.add_trace(
        go.Scatter(x=discount_df['Discount'], y=discount_df['net_sale'],
                  mode='markers', name='Discount vs Revenue',
                  marker=dict(size=6, color='#d62728', opacity=0.6)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Discount Performance Analysis")
    return fig

def create_sales_predictions(df):
    """Create sales predictions using machine learning"""
    if df.empty or len(df) < 30:
        st.warning("Insufficient data for predictions")
        return go.Figure(), {}
    
    try:
        # Prepare data for prediction
        daily_sales = df.groupby('Date').agg({
            'net_sale': 'sum',
            'OrderID': 'count',
            'Discount': 'sum'
        }).reset_index()
        
        daily_sales['Date_ordinal'] = pd.to_datetime(daily_sales['Date']).map(datetime.toordinal)
        daily_sales['DayOfWeek'] = pd.to_datetime(daily_sales['Date']).dt.dayofweek
        daily_sales['Month'] = pd.to_datetime(daily_sales['Date']).dt.month
        
        # Features for prediction
        features = ['Date_ordinal', 'DayOfWeek', 'Month', 'OrderID', 'Discount']
        X = daily_sales[features]
        y = daily_sales['net_sale']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        # Model performance
        lr_mae = mean_absolute_error(y_test, lr_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # Future predictions
        last_date = daily_sales['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        future_data = pd.DataFrame({
            'Date': future_dates,
            'Date_ordinal': future_dates.map(datetime.toordinal),
            'DayOfWeek': future_dates.dayofweek,
            'Month': future_dates.month,
            'OrderID': daily_sales['OrderID'].mean(),  # Use average
            'Discount': daily_sales['Discount'].mean()
        })
        
        future_lr_pred = lr_model.predict(future_data[features])
        future_rf_pred = rf_model.predict(future_data[features])
        
        # Visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_sales['Date'], y=daily_sales['net_sale'],
            mode='lines+markers', name='Historical Sales',
            line=dict(color='#1f77b4')
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=future_data['Date'], y=future_lr_pred,
            mode='lines', name='Linear Regression Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_data['Date'], y=future_rf_pred,
            mode='lines', name='Random Forest Forecast',
            line=dict(color='#2ca02c', dash='dot')
        ))
        
        fig.update_layout(
            title='Sales Predictions (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Revenue',
            height=500
        )
        
        model_metrics = {
            'Linear Regression': {'MAE': lr_mae, 'R²': lr_r2},
            'Random Forest': {'MAE': rf_mae, 'R²': rf_r2}
        }
        
        return fig, model_metrics
        
    except Exception as e:
        st.error(f"Error creating predictions: {e}")
        return go.Figure(), {}

def create_comparison_analysis(df):
    """Create comparative analysis across different dimensions"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales by Channel', 'Sales by Location', 
                       'Payment Method Analysis', 'Brand Performance'),
        specs=[[{"type": "pie"}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sales by Channel
    if 'Channel' in df.columns:
        channel_sales = df.groupby('Channel')['net_sale'].sum()
        fig.add_trace(
            go.Pie(labels=channel_sales.index, values=channel_sales.values,
                   name="Channel Sales"),
            row=1, col=1
        )
    
    # Sales by Location
    if 'Location' in df.columns:
        location_sales = df.groupby('Location')['net_sale'].sum().nlargest(10)
        fig.add_trace(
            go.Bar(x=location_sales.index, y=location_sales.values,
                   name='Location Sales', marker_color='#ff7f0e'),
            row=1, col=2
        )
    
    # Payment Method Analysis
    if 'PaymentMethod' in df.columns:
        payment_sales = df.groupby('PaymentMethod')['net_sale'].sum()
        fig.add_trace(
            go.Bar(x=payment_sales.index, y=payment_sales.values,
                   name='Payment Method', marker_color='#2ca02c'),
            row=2, col=1
        )
    
    # Brand Performance
    if 'Brand' in df.columns:
        brand_sales = df.groupby('Brand')['net_sale'].sum()
        fig.add_trace(
            go.Bar(x=brand_sales.index, y=brand_sales.values,
                   name='Brand Performance', marker_color='#d62728'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True, title_text="Comparative Sales Analysis")
    return fig

def create_pivot_table_analysis(df):
    """Create interactive pivot table analysis"""
    if df.empty:
        st.warning("No data available for pivot table analysis")
        return
    
    st.markdown('<div class="section-header">📊 Interactive Pivot Table Analysis</div>', unsafe_allow_html=True)
    
    # Prediction Section at the top
    st.subheader("🔮 Today's Predictions")
    
    try:
        # Generate predictions for today
        current_date = datetime.now().date()
        
        # Get historical data for prediction (last 30 days)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            historical_data = df[df['Date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
            
            if len(historical_data) >= 7:  # Need at least a week of data
                # Calculate daily averages for prediction
                daily_stats = historical_data.groupby('Date').agg({
                    'net_sale': 'sum',
                    'Discount': 'sum',
                    'OrderID': 'count'
                }).reset_index()
                
                # Calculate trends
                avg_net_sale = daily_stats['net_sale'].mean()
                avg_discount = daily_stats['Discount'].mean()
                avg_orders = daily_stats['OrderID'].mean()
                
                # Simple trend calculation (last 7 days vs previous 7 days)
                if len(daily_stats) >= 14:
                    recent_avg = daily_stats.tail(7)['net_sale'].mean()
                    previous_avg = daily_stats.iloc[-14:-7]['net_sale'].mean()
                    trend_factor = recent_avg / previous_avg if previous_avg > 0 else 1.0
                else:
                    trend_factor = 1.0
                
                # Apply trend to predictions
                predicted_net_sale = avg_net_sale * trend_factor
                predicted_discount = avg_discount * trend_factor
                predicted_orders = avg_orders * trend_factor
                predicted_discount_pct = (predicted_discount / (predicted_net_sale + predicted_discount) * 100) if (predicted_net_sale + predicted_discount) > 0 else 0
                
                # Display predictions in columns
                pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                
                with pred_col1:
                    st.metric(
                        label="🎯 Predicted Net Sales",
                        value=f"{predicted_net_sale:,.0f}",
                        delta=f"{((trend_factor - 1) * 100):+.1f}%" if trend_factor != 1.0 else None
                    )
                
                with pred_col2:
                    st.metric(
                        label="💰 Predicted Discount",
                        value=f"{predicted_discount:,.0f}",
                        delta=f"{((trend_factor - 1) * 100):+.1f}%" if trend_factor != 1.0 else None
                    )
                
                with pred_col3:
                    st.metric(
                        label="📈 Predicted Orders",
                        value=f"{predicted_orders:,.0f}",
                        delta=f"{((trend_factor - 1) * 100):+.1f}%" if trend_factor != 1.0 else None
                    )
                
                with pred_col4:
                    st.metric(
                        label="🎯 Predicted Discount %",
                        value=f"{predicted_discount_pct:.1f}%",
                        delta=None
                    )
                
                st.info(f"📊 **Predictions based on**: {len(daily_stats)} days of historical data with {trend_factor:.2f}x trend factor")
            else:
                st.warning("⚠️ Insufficient historical data for predictions (need at least 7 days)")
    
    except Exception as e:
        st.warning(f"⚠️ Unable to generate predictions: {str(e)}")
    
    st.divider()
    
    # Add custom calculated columns for pivot table
    if 'net_sale' in df.columns and 'Discount' in df.columns and 'GrossPrice' in df.columns:
        # Ensure Discount_Percentage is calculated from GrossPrice (already done in add_calculated_columns, but ensure consistency)
        df['Discount_Percentage'] = (df['Discount'] / df['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
    
    # Date filtering section
    st.subheader("📅 Date Filter")
    
    # Convert Date column to datetime if it's not already
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Smart default: Latest date and latest date - 1
        latest_date = df['Date'].max().date()
        second_latest_date = (df['Date'].max() - pd.Timedelta(days=1)).date()
        
        # Ensure second_latest_date doesn't go below minimum available date
        min_available_date = df['Date'].min().date()
        if second_latest_date < min_available_date:
            second_latest_date = min_available_date
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=second_latest_date,  # Default: latest date - 1
                min_value=df['Date'].min().date(),
                max_value=df['Date'].max().date(),
                help="Default: Latest date - 1 for comparative analysis"
            )
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=latest_date,  # Default: latest date
                min_value=df['Date'].min().date(),
                max_value=df['Date'].max().date(),
                help="Default: Latest date available in data"
            )
        
        # Show smart default info
        if start_date == second_latest_date and end_date == latest_date:
            st.info(f"📊 **Smart Default**: Comparing {second_latest_date} vs {latest_date} (2 most recent dates)")
        
        # Apply date filter
        df_filtered = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)].copy()
        
        # Additional filtering info
        excluded_info = ""
        if 'Discount_Percentage' in df.columns:
            total_orders = len(df)
            filtered_orders = len(df_filtered)
            hundred_percent_excluded = len(df[df['Discount_Percentage'] >= 100.0])
            if hundred_percent_excluded > 0:
                excluded_info = f" (excluding {hundred_percent_excluded:,} orders with 100% discount)"
        
        st.info(f"📊 Filtered data: {len(df_filtered):,} records from {start_date} to {end_date}{excluded_info}")
    else:
        df_filtered = df.copy()
        st.warning("⚠️ Date column not found. Using all data.")
    
    # Apply synchronized time filter silently (background processing)
    if 'ReceivedAt' in df_filtered.columns and len(df_filtered) > 0:
        # Find latest timestamp for synchronization
        latest_timestamp = df_filtered['ReceivedAt'].max()
        latest_hour = latest_timestamp.hour
        latest_minute = latest_timestamp.minute
        latest_date = latest_timestamp.date()
        
        # Apply synchronized time filter to both dates (up to exact minute of latest order)
        df_filtered = df_filtered[df_filtered['ReceivedAt'].dt.time <= latest_timestamp.time()].copy()
        df_filtered['Hour'] = df_filtered['ReceivedAt'].dt.hour
        
        # Store sync info for display in pivot configuration
        sync_info = {
            'latest_timestamp': latest_timestamp,
            'latest_hour': latest_hour,
            'latest_minute': latest_minute,
            'latest_date': latest_date
        }
    else:
        sync_info = None
    
    # Sidebar controls for pivot table configuration
    st.subheader("🔧 Pivot Table Configuration")
    
    # Hour Filter Section
    selected_max_hour = None  # Initialize variable to avoid NameError
    if sync_info:
        st.markdown("### ⏰ Hour Filter")
        
        # Display sync information
        st.info(f"🕐 **Latest Order**: {sync_info['latest_timestamp'].strftime('%Y-%m-%d %H:%M')} (auto-synchronized)")
        
        # Get available hours in the synchronized data
        available_hours = sorted(df_filtered['Hour'].unique()) if 'Hour' in df_filtered.columns else []
        
        if available_hours:
            col1, col2 = st.columns(2)
            
            with col1:
                # Hour range selection
                max_hour = sync_info['latest_hour']
                selected_max_hour = st.selectbox(
                    "📊 Filter to Hour:",
                    options=list(range(max_hour + 1)),
                    index=max_hour,
                    format_func=lambda x: f"Up to {x:02d}:xx ({x+1} hours)",
                    help=f"Select maximum hour to include. Data is auto-synced up to {sync_info['latest_timestamp'].strftime('%H:%M')}"
                )
            
            with col2:
                st.write(f"**📋 Current Selection:**")
                st.write(f"• **Hours**: 0-{selected_max_hour}")
                st.write(f"• **Time Range**: 00:00 to {selected_max_hour:02d}:59")
                if selected_max_hour == max_hour:
                    st.write(f"• **Exact Sync**: Up to {sync_info['latest_timestamp'].strftime('%H:%M')}")
            
            # Apply hour filter to the data
            df_filtered = df_filtered[df_filtered['Hour'] <= selected_max_hour].copy()
            
            # Show filter results
            st.success(f"✅ **Hour Filter Applied**: {len(df_filtered):,} records (Hours 0-{selected_max_hour})")
        else:
            st.warning("⚠️ No hour data available for filtering")
            selected_max_hour = None  # Ensure it's explicitly None when no hours available
    
    # Available columns for pivot table
    numeric_columns = [
        'net_sale', 'GrossPrice', 'Discount', 'Tips', 'Delivery', 'VAT', 'Surcharge', 'Total',
        'Profit_Margin', 'Discount_Percentage', 'Order_Profitability', 'Revenue_After_Delivery'
    ]
    
    categorical_columns = [
        'CustomerName', 'DayOfWeek', 'Month', 'Quarter', 'Hour', 'Date',
        'Order_Size_Category', 'Discount_Category', 'Time_Period'
    ]
    
    # Add conditional columns if they exist
    if 'Channel' in df_filtered.columns:
        categorical_columns.append('Channel')
    if 'Brand' in df_filtered.columns:
        categorical_columns.append('Brand')
    if 'Location' in df_filtered.columns:
        categorical_columns.append('Location')
    if 'PaymentMethod' in df_filtered.columns:
        categorical_columns.append('PaymentMethod')
    if 'Customer_Value_Tier' in df_filtered.columns:
        categorical_columns.append('Customer_Value_Tier')
    
    # Filter columns that actually exist in the dataframe
    available_numeric = [col for col in numeric_columns if col in df_filtered.columns]
    available_categorical = [col for col in categorical_columns if col in df_filtered.columns]
    
    # Quick Setup Section for your specific requirement
    st.subheader("⚡ Quick Setup: Channel Analysis")
    
    use_quick_setup = st.checkbox("Use Quick Setup (Channel × Date with Net Sale, Discount, Discount%)", value=True)
    
    if use_quick_setup and 'Channel' in df_filtered.columns:
        # Quick setup configuration
        rows = ['Channel']
        columns = ['Date']
        
        # Create the combined pivot table with multiple values
        try:
            # Check if we have enough data
            if len(df_filtered) == 0:
                st.warning("⚠️ No data available for the selected filters. Please adjust your date/time filters.")
                return
            
            # Prepare data for multi-value pivot
            pivot_data = df_filtered.groupby(['Channel', 'Date']).agg({
                'net_sale': 'sum',
                'Discount': 'sum',
                'GrossPrice': 'sum',
                'OrderID': 'count'
            }).reset_index()
            
            # Check if pivot_data has any rows
            if len(pivot_data) == 0:
                st.warning("⚠️ No data found for Channel and Date grouping. Please check your data.")
                return
            
            # Calculate Discount Percentage (from GrossPrice, not net_sale)
            pivot_data['Discount_Percentage'] = (pivot_data['Discount'] / pivot_data['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
            
            # Create individual pivot tables
            pivot_net_sale = pivot_data.pivot_table(
                values='net_sale',
                index='Channel',
                columns='Date',
                aggfunc='sum',
                fill_value=0
            )
            
            pivot_discount = pivot_data.pivot_table(
                values='Discount',
                index='Channel',
                columns='Date',
                aggfunc='sum',
                fill_value=0
            )
            
            pivot_discount_pct = pivot_data.pivot_table(
                values='Discount_Percentage',
                index='Channel',
                columns='Date',
                aggfunc='mean',
                fill_value=0
            )
            
            # Create the combined multi-level pivot table
            combined_pivot = pd.DataFrame()
            
            # Get all unique dates for column ordering
            dates = sorted(pivot_net_sale.columns)
            
            # Create multi-level columns: (Date, Metric)
            multi_cols = []
            for date in dates:
                multi_cols.extend([
                    (str(date), 'Net Sale'),
                    (str(date), 'Discount'),
                    (str(date), 'Discount %')
                ])
            
            # Create the combined dataframe
            combined_data = {}
            for date in dates:
                date_str = str(date)
                combined_data[(date_str, 'Net Sale')] = pivot_net_sale[date] if date in pivot_net_sale.columns else 0
                combined_data[(date_str, 'Discount')] = pivot_discount[date] if date in pivot_discount.columns else 0
                combined_data[(date_str, 'Discount %')] = pivot_discount_pct[date] if date in pivot_discount_pct.columns else 0
            
            combined_pivot = pd.DataFrame(combined_data, index=pivot_net_sale.index)
            
            # Create multi-index columns
            combined_pivot.columns = pd.MultiIndex.from_tuples(combined_pivot.columns, names=['Date', 'Metric'])
            
            # Add totals row
            totals_row = {}
            for date in dates:
                date_str = str(date)
                net_sale_total = combined_pivot[(date_str, 'Net Sale')].sum()
                discount_total = combined_pivot[(date_str, 'Discount')].sum()
                discount_pct_avg = (discount_total / net_sale_total * 100) if net_sale_total > 0 else 0
                
                totals_row[(date_str, 'Net Sale')] = net_sale_total
                totals_row[(date_str, 'Discount')] = discount_total
                totals_row[(date_str, 'Discount %')] = discount_pct_avg
            
            # Add totals row to combined pivot
            totals_df = pd.DataFrame([totals_row], columns=combined_pivot.columns, index=['Total'])
            combined_pivot = pd.concat([combined_pivot, totals_df])
            
            # Add difference columns if we have exactly 2 dates
            if len(dates) == 2:
                # Ensure dates are sorted (latest first)
                sorted_dates = sorted(dates, reverse=True)
                latest_date = str(sorted_dates[0])
                second_date = str(sorted_dates[1])
                
                # Calculate differences: Latest - Second
                combined_pivot[(f'Δ {latest_date} vs {second_date}', 'Net Sale Δ')] = (
                    combined_pivot[(latest_date, 'Net Sale')] - combined_pivot[(second_date, 'Net Sale')]
                )
                
                combined_pivot[(f'Δ {latest_date} vs {second_date}', 'Discount Δ')] = (
                    combined_pivot[(latest_date, 'Discount')] - combined_pivot[(second_date, 'Discount')]
                )
                
                combined_pivot[(f'Δ {latest_date} vs {second_date}', 'Discount % Δ')] = (
                    combined_pivot[(latest_date, 'Discount %')] - combined_pivot[(second_date, 'Discount %')]
                )
                
                # Recreate columns to maintain proper order
                new_columns = []
                for date in sorted_dates:
                    date_str = str(date)
                    new_columns.extend([
                        (date_str, 'Net Sale'),
                        (date_str, 'Discount'),
                        (date_str, 'Discount %')
                    ])
                
                # Add difference columns at the end
                new_columns.extend([
                    (f'Δ {latest_date} vs {second_date}', 'Net Sale Δ'),
                    (f'Δ {latest_date} vs {second_date}', 'Discount Δ'),
                    (f'Δ {latest_date} vs {second_date}', 'Discount % Δ')
                ])
                
                # Reorder columns
                combined_pivot = combined_pivot[new_columns]
            
            # Display configuration summary
            if sync_info and selected_max_hour is not None:
                if selected_max_hour == sync_info['latest_hour']:
                    time_info = f" | Up to {sync_info['latest_timestamp'].strftime('%H:%M')} (exact sync)"
                else:
                    time_info = f" | Hours 0-{selected_max_hour}"
            else:
                time_info = ""
            st.info(f"📊 **Combined Pivot Table**: Channel × Date with Net Sale, Discount & Discount% | **Period**: {start_date} to {end_date}{time_info}")
            
            # Display the combined pivot table
            st.subheader("📊 Combined Sales Analysis: Channel × Date")
            if len(dates) == 2:
                st.markdown("**Multi-metric view with Net Sale, Discount amounts, Discount percentages, and day-to-day differences**")
            else:
                st.markdown("**Multi-metric view with Net Sale, Discount amounts, and Discount percentages**")
            
            # Format the dataframe for better display
            formatted_combined = combined_pivot.copy()
            
            # Round numeric values appropriately
            for col in formatted_combined.columns:
                if 'Discount %' in col[1] or 'Discount % Δ' in col[1]:
                    formatted_combined[col] = formatted_combined[col].round(1)
                else:
                    formatted_combined[col] = formatted_combined[col].round(2)
            
            # Create custom styling for difference columns
            def style_differences(val):
                if pd.isna(val):
                    return ''
                # For discount differences: reduction (negative) should be green, increase (positive) should be red
                # For net sale differences: increase (positive) should be green, decrease (negative) should be red
                if val > 0:
                    return 'background-color: #d4edda; color: #155724'  # Green for positive (good for net sales)
                elif val < 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Red for negative (bad for net sales)
                else:
                    return 'background-color: #fff3cd; color: #856404'  # Yellow for zero
            
            def style_discount_differences(val):
                if pd.isna(val):
                    return ''
                # For discount differences: reduction (negative) should be green, increase (positive) should be red
                if val > 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Red for positive (bad - more discount)
                elif val < 0:
                    return 'background-color: #d4edda; color: #155724'  # Green for negative (good - less discount)
                else:
                    return 'background-color: #fff3cd; color: #856404'  # Yellow for zero
            
            # Apply styling
            styled_df = formatted_combined.style.format({
                col: '{:.2f}' if 'Discount %' not in col[1] else '{:.1f}%' 
                for col in formatted_combined.columns
            })
            
            # Apply difference column styling if they exist
            if len(dates) == 2:
                diff_cols = [col for col in formatted_combined.columns if 'Δ' in col[0]]
                for col in diff_cols:
                    if 'Discount Δ' in col[1] or 'Discount % Δ' in col[1]:
                        # For discount metrics: reduction (negative) = green, increase (positive) = red
                        styled_df = styled_df.applymap(style_discount_differences, subset=[col])
                    else:
                        # For net sale metrics: increase (positive) = green, decrease (negative) = red
                        styled_df = styled_df.applymap(style_differences, subset=[col])
            
            # Display with custom styling
            st.dataframe(styled_df, use_container_width=True)
            
            # Download option for Combined Table
            if sync_info and selected_max_hour is not None:
                if selected_max_hour == sync_info['latest_hour']:
                    time_suffix = f"_up_to_{sync_info['latest_hour']:02d}{sync_info['latest_minute']:02d}"
                else:
                    time_suffix = f"_h0-{selected_max_hour}"
            else:
                time_suffix = ""
            
            csv_combined = combined_pivot.to_csv()
            st.download_button(
                label="📥 Download Combined Pivot Table CSV",
                data=csv_combined,
                file_name=f"combined_pivot_channel_date_{start_date}_{end_date}{time_suffix}.csv",
                mime="text/csv",
                key="download_combined"
            )
            
            # Additional download options for individual metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_net_sale = pivot_net_sale.to_csv()
                st.download_button(
                    label="📥 Net Sale Only",
                    data=csv_net_sale,
                    file_name=f"net_sale_only_{start_date}_{end_date}{time_suffix}.csv",
                    mime="text/csv",
                    key="download_net_sale_only"
                )
            
            with col2:
                csv_discount = pivot_discount to_csv()
                st.download_button(
                    label="📥 Discount Only",
                    data=csv_discount,
                    file_name=f"discount_only_{start_date}_{end_date}{time_suffix}.csv",
                    mime="text/csv",
                    key="download_discount_only"
                )
            
            with col3:
                csv_discount_pct = pivot_discount_pct.to_csv()
                st.download_button(
                    label="📥 Discount % Only",
                    data=csv_discount_pct,
                    file_name=f"discount_pct_only_{start_date}_{end_date}{time_suffix}.csv",
                    mime="text/csv",
                    key="download_discount_pct_only"
                )
            
            # Visualization of combined pivot table
            st.subheader("📈 Combined Pivot Table Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Net Sale Heatmap (extract from combined table)
                net_sale_cols = [col for col in combined_pivot.columns if col[1] == 'Net Sale']
                if len(net_sale_cols) > 1 and len(combined_pivot.index) > 1:
                    # Create net sale only dataframe for heatmap
                    net_sale_data = combined_pivot[net_sale_cols].iloc[:-1]  # Remove totals row
                    net_sale_data.columns = [col[0] for col in net_sale_data.columns]  # Flatten column names
                    
                    fig_heatmap = px.imshow(
                        net_sale_data,
                        title="Net Sale Heatmap: Channel × Date",
                        labels=dict(x="Date", y="Channel", color="Net Sale"),
                        aspect="auto",
                        color_continuous_scale="Blues"
                    )
                    fig_heatmap.update_layout(height=400)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.write("📊 Insufficient data for net sale heatmap")
            
            with viz_col2:
                # Discount Percentage Heatmap (extract from combined table)
                discount_pct_cols = [col for col in combined_pivot.columns if col[1] == 'Discount %']
                if len(discount_pct_cols) > 1 and len(combined_pivot.index) > 1:
                    # Create discount % only dataframe for heatmap
                    discount_pct_data = combined_pivot[discount_pct_cols].iloc[:-1]  # Remove totals row
                    discount_pct_data.columns = [col[0] for col in discount_pct_data.columns]  # Flatten column names
                    
                    fig_heatmap_pct = px.imshow(
                        discount_pct_data,
                        title="Discount % Heatmap: Channel × Date",
                        labels=dict(x="Date", y="Channel", color="Discount %"),
                        aspect="auto",
                        color_continuous_scale="Reds"
                    )
                    fig_heatmap_pct.update_layout(height=400)
                    st.plotly_chart(fig_heatmap_pct, use_container_width=True)
                else:
                    st.write("📊 Insufficient data for discount % heatmap")
            
            # Top 10 / Bottom 10 Analysis Section
            st.subheader("🏆 Top 10 & Bottom 10 Performance Analysis")
            
            try:
                # Create analysis for Brand, Location, and Channel
                analysis_dimensions = []
                if 'Brand' in df_filtered.columns:
                    analysis_dimensions.append('Brand')
                if 'Location' in df_filtered.columns:
                    analysis_dimensions.append('Location')
                if 'Channel' in df_filtered.columns:
                    analysis_dimensions.append('Channel')
                
                if analysis_dimensions:
                    # Let user select which dimension to analyze
                    selected_dimension = st.selectbox(
                        "📊 Select Analysis Dimension:",
                        options=analysis_dimensions,
                        index=0,
                        help="Choose which dimension to analyze for top/bottom performance"
                    )
                    
                    if selected_dimension in df_filtered.columns:
                        # Create analysis data with same filters as pivot table
                        analysis_data = df_filtered.groupby([selected_dimension, 'Date']).agg({
                            'net_sale': 'sum',
                            'Discount': 'sum',
                            'GrossPrice': 'sum',
                            'OrderID': 'count'
                        }).reset_index()
                        
                        # Calculate Discount Percentage
                        analysis_data['Discount_Percentage'] = (analysis_data['Discount'] / analysis_data['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
                        
                        # Create pivot for analysis
                        analysis_pivot_net = analysis_data.pivot_table(
                            values='net_sale',
                            index=selected_dimension,
                            columns='Date',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        analysis_pivot_discount = analysis_data.pivot_table(
                            values='Discount',
                            index=selected_dimension,
                            columns='Date',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        analysis_pivot_discount_pct = analysis_data.pivot_table(
                            values='Discount_Percentage',
                            index=selected_dimension,
                            columns='Date',
                            aggfunc='mean',
                            fill_value=0
                        )
                        
                        # Create combined analysis table with differences (if 2 dates)
                        analysis_dates = sorted(analysis_pivot_net.columns)
                        
                        # Create summary table
                        summary_data = []
                        for item in analysis_pivot_net.index:
                            row_data = {'Item': item}
                            
                            # Add data for each date
                            for date in analysis_dates:
                                date_str = str(date)
                                row_data[f'{date_str}_Net_Sale'] = analysis_pivot_net.loc[item, date] if date in analysis_pivot_net.columns else 0
                                row_data[f'{date_str}_Discount'] = analysis_pivot_discount.loc[item, date] if date in analysis_pivot_discount.columns else 0
                                row_data[f'{date_str}_Discount_Pct'] = analysis_pivot_discount_pct.loc[item, date] if date in analysis_pivot_discount_pct.columns else 0
                            
                            # Add differences if exactly 2 dates
                            if len(analysis_dates) == 2:
                                latest_date = analysis_dates[-1]  # Most recent
                                previous_date = analysis_dates[0]  # Previous
                                
                                row_data['Net_Sale_Diff'] = row_data[f'{latest_date}_Net_Sale'] - row_data[f'{previous_date}_Net_Sale']
                                row_data['Discount_Diff'] = row_data[f'{latest_date}_Discount'] - row_data[f'{previous_date}_Discount']
                                row_data['Discount_Pct_Diff'] = row_data[f'{latest_date}_Discount_Pct'] - row_data[f'{previous_date}_Discount_Pct']
                            
                            # Calculate total for ranking
                            row_data['Total_Net_Sale'] = sum([row_data[f'{date}_Net_Sale'] for date in [str(d) for d in analysis_dates]])
                            
                            summary_data.append(row_data)
                        
                        # Convert to DataFrame
                        summary_df = pd.DataFrame(summary_data)
                        
                        if len(summary_df) > 0:
                            # Sort by total net sales for ranking
                            summary_df_sorted = summary_df.sort_values('Total_Net_Sale', ascending=False)
                            
                            # Get top 10 and bottom 10
                            top_10 = summary_df_sorted.head(10).copy()
                            bottom_10 = summary_df_sorted.tail(10).copy()
                            
                            # Display tables side by side
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"### 🏆 Top 10 {selected_dimension}s")
                                
                                # Format top 10 for display
                                display_top = top_10[['Item'] + [col for col in top_10.columns if col not in ['Item', 'Total_Net_Sale']]].copy()
                                
                                # Round numeric columns
                                for col in display_top.columns:
                                    if col != 'Item' and display_top[col].dtype in ['float64', 'int64']:
                                        if 'Discount_Pct' in col:
                                            display_top[col] = display_top[col].round(1)
                                        else:
                                            display_top[col] = display_top[col].round(2)
                                
                                # Apply conditional formatting for differences
                                if len(analysis_dates) == 2:
                                    styled_top = display_top.style.format({
                                        col: '{:.2f}' if 'Discount_Pct' not in col else '{:.1f}%'
                                        for col in display_top.columns if col != 'Item'
                                    })
                                    
                                    # Apply styling to difference columns
                                    diff_cols = [col for col in display_top.columns if '_Diff' in col]
                                    for col in diff_cols:
                                        if 'Discount' in col:
                                            styled_top = styled_top.applymap(style_discount_differences, subset=[col])
                                        else:
                                            styled_top = styled_top.applymap(style_differences, subset=[col])
                                    
                                    st.dataframe(styled_top, use_container_width=True)
                                else:
                                    st.dataframe(display_top, use_container_width=True)
                            
                            with col2:
                                st.markdown(f"### 📉 Bottom 10 {selected_dimension}s")
                                
                                # Format bottom 10 for display
                                display_bottom = bottom_10[['Item'] + [col for col in bottom_10.columns if col not in ['Item', 'Total_Net_Sale']]].copy()
                                
                                # Round numeric columns
                                for col in display_bottom.columns:
                                    if col != 'Item' and display_bottom[col].dtype in ['float64', 'int64']:
                                        if 'Discount_Pct' in col:
                                            display_bottom[col] = display_bottom[col].round(1)
                                        else:
                                            display_bottom[col] = display_bottom[col].round(2)
                                
                                # Apply conditional formatting for differences
                                if len(analysis_dates) == 2:
                                    styled_bottom = display_bottom.style.format({
                                        col: '{:.2f}' if 'Discount_Pct' not in col else '{:.1f}%'
                                        for col in display_bottom.columns if col != 'Item'
                                    })
                                    
                                    # Apply styling to difference columns
                                    diff_cols = [col for col in display_bottom.columns if '_Diff' in col]
                                    for col in diff_cols:
                                        if 'Discount' in col:
                                            styled_bottom = styled_bottom.applymap(style_discount_differences, subset=[col])
                                        else:
                                            styled_bottom = styled_bottom.applymap(style_differences, subset=[col])
                                    
                                    st.dataframe(styled_bottom, use_container_width=True)
                                else:
                                    st.dataframe(display_bottom, use_container_width=True)
                            
                            # Download options for top/bottom analysis
                            dl_col1, dl_col2 = st.columns(2)
                            
                            with dl_col1:
                                if sync_info and selected_max_hour is not None:
                                    if selected_max_hour == sync_info['latest_hour']:
                                        time_suffix = f"_up_to_{sync_info['latest_hour']:02d}{sync_info['latest_minute']:02d}"
                                    else:
                                        time_suffix = f"_h0-{selected_max_hour}"
                                else:
                                    time_suffix = ""
                                
                                csv_top_10 = display_top.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download Top 10 {selected_dimension}s",
                                    data=csv_top_10,
                                    file_name=f"top_10_{selected_dimension.lower()}_{start_date}_{end_date}{time_suffix}.csv",
                                    mime="text/csv",
                                    key="download_top_10"
                                )
                            
                            with dl_col2:
                                csv_bottom_10 = display_bottom.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download Bottom 10 {selected_dimension}s",
                                    data=csv_bottom_10,
                                    file_name=f"bottom_10_{selected_dimension.lower()}_{start_date}_{end_date}{time_suffix}.csv",
                                    mime="text/csv",
                                    key="download_bottom_10"
                                )
                            
                            # Quick insights for top/bottom analysis
                            st.markdown("### 💡 Top/Bottom Analysis Insights")
                            
                            insight_col1, insight_col2, insight_col3 = st.columns(3)
                            
                            with insight_col1:
                                best_performer = top_10.iloc[0]['Item']
                                best_sales = top_10.iloc[0]['Total_Net_Sale']
                                st.metric(
                                    label=f"🥇 Best {selected_dimension}",
                                    value=best_performer,
                                    delta=f"{best_sales:,.0f} total sales"
                                )
                            
                            with insight_col2:
                                worst_performer = bottom_10.iloc[-1]['Item']
                                worst_sales = bottom_10.iloc[-1]['Total_Net_Sale']
                                st.metric(
                                    label=f"📉 Lowest {selected_dimension}",
                                    value=worst_performer,
                                    delta=f"{worst_sales:,.0f} total sales"
                                )
                            
                            with insight_col3:
                                performance_gap = best_sales - worst_sales
                                st.metric(
                                    label="📏 Performance Gap",
                                    value=f"{performance_gap:,.0f}",
                                    delta="Sales difference (Best - Worst)"
                                )
                            
                            # Day-to-day insights for top/bottom if 2 dates
                            if len(analysis_dates) == 2:
                                st.write("📊 **Day-to-Day Performance Changes:**")
                                
                                # Best improvement in top 10
                                if 'Net_Sale_Diff' in top_10.columns:
                                    best_improvement = top_10.loc[top_10['Net_Sale_Diff'].idxmax()]
                                    worst_decline_top = top_10.loc[top_10['Net_Sale_Diff'].idxmin()]
                                    
                                    st.write(f"• **Best Improvement (Top 10)**: {best_improvement['Item']} (+{best_improvement['Net_Sale_Diff']:,.0f})")
                                    st.write(f"• **Biggest Decline (Top 10)**: {worst_decline_top['Item']} ({worst_decline_top['Net_Sale_Diff']:,.0f})")
                        
                        else:
                            st.warning(f"⚠️ No data available for {selected_dimension} analysis")
                
                else:
                    st.warning("⚠️ No suitable dimensions (Brand, Location, Channel) found for top/bottom analysis")
            
            except Exception as e:
                st.error(f"❌ Error creating top/bottom analysis: {str(e)}")
            
            # Comprehensive Decline Analysis across all dimensions
            if len(dates) == 2:
                st.subheader("🔍 Comprehensive Decline Analysis - Combined Dimensions")
                st.markdown("**Identify exactly where performance drops are occurring across Location + Brand + Channel combinations**")
                
                try:
                    # Determine which dimensions are available
                    available_dimensions = []
                    if 'Location' in df_filtered.columns:
                        available_dimensions.append('Location')
                    if 'Brand' in df_filtered.columns:
                        available_dimensions.append('Brand')
                    if 'Channel' in df_filtered.columns:
                        available_dimensions.append('Channel')
                    
                    if len(available_dimensions) > 0:
                        # Create combined analysis with all available dimensions
                        combined_analysis = df_filtered.groupby(available_dimensions + ['Date']).agg({
                            'net_sale': 'sum',
                            'Discount': 'sum',
                            'GrossPrice': 'sum',
                            'OrderID': 'count'
                        }).reset_index()
                        
                        # Calculate Discount Percentage
                        combined_analysis['Discount_Percentage'] = (combined_analysis['Discount'] / combined_analysis['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
                        
                        # Create pivot for combined dimensions
                        combined_pivot_net = combined_analysis.pivot_table(
                            values='net_sale',
                            index=available_dimensions,
                            columns='Date',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        combined_pivot_discount = combined_analysis.pivot_table(
                            values='Discount',
                            index=available_dimensions,
                            columns='Date',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        combined_pivot_discount_pct = combined_analysis.pivot_table(
                            values='Discount_Percentage',
                            index=available_dimensions,
                            columns='Date',
                            aggfunc='mean',
                            fill_value=0
                        )
                        
                        # Calculate differences for combined dimensions
                        combined_dates = sorted(combined_pivot_net.columns)
                        all_combined_data = []
                        
                        if len(combined_dates) == 2:
                            latest_date = combined_dates[-1]
                            previous_date = combined_dates[0]
                            
                            for item_tuple in combined_pivot_net.index:
                                # Handle both single and multi-index cases
                                if isinstance(item_tuple, tuple):
                                    item_dict = {dim: item_tuple[i] for i, dim in enumerate(available_dimensions)}
                                    item_display = " | ".join([f"{dim}: {item_tuple[i]}" for i, dim in enumerate(available_dimensions)])
                                else:
                                    item_dict = {available_dimensions[0]: item_tuple}
                                    item_display = f"{available_dimensions[0]}: {item_tuple}"
                                
                                net_sale_latest = combined_pivot_net.loc[item_tuple, latest_date] if latest_date in combined_pivot_net.columns else 0
                                net_sale_previous = combined_pivot_net.loc[item_tuple, previous_date] if previous_date in combined_pivot_net.columns else 0
                                net_sale_diff = net_sale_latest - net_sale_previous
                                net_sale_pct_change = (net_sale_diff / net_sale_previous * 100) if net_sale_previous > 0 else 0
                                
                                discount_latest = combined_pivot_discount.loc[item_tuple, latest_date] if latest_date in combined_pivot_discount.columns else 0
                                discount_previous = combined_pivot_discount.loc[item_tuple, previous_date] if previous_date in combined_pivot_discount.columns else 0
                                discount_diff = discount_latest - discount_previous
                                
                                discount_pct_latest = combined_pivot_discount_pct.loc[item_tuple, latest_date] if latest_date in combined_pivot_discount_pct.columns else 0
                                discount_pct_previous = combined_pivot_discount_pct.loc[item_tuple, previous_date] if previous_date in combined_pivot_discount_pct.columns else 0
                                discount_pct_diff = discount_pct_latest - discount_pct_previous
                                
                                # Create row data
                                row_data = {
                                    'Combined_Dimensions': item_display,
                                    'Previous_Net_Sale': net_sale_previous,
                                    'Latest_Net_Sale': net_sale_latest,
                                    'Net_Sale_Diff': net_sale_diff,
                                    'Net_Sale_Pct_Change': net_sale_pct_change,
                                    'Previous_Discount': discount_previous,
                                    'Latest_Discount': discount_latest,
                                    'Discount_Diff': discount_diff,
                                    'Previous_Discount_Pct': discount_pct_previous,
                                    'Latest_Discount_Pct': discount_pct_latest,
                                    'Discount_Pct_Diff': discount_pct_diff
                                }
                                
                                # Add individual dimension data for filtering
                                row_data.update(item_dict)
                                
                                all_combined_data.append(row_data)
                        
                        if all_combined_data:
                            # Convert to DataFrame
                            combined_decline_df = pd.DataFrame(all_combined_data)
                            
                            # Create analysis tabs for different views
                            decline_tab1, decline_tab2, decline_tab3 = st.tabs([
                                "📉 Biggest Declines", "📈 Biggest Improvements", "🔄 All Changes"
                            ])
                            
                            with decline_tab1:
                                st.markdown("### 📉 Biggest Declines - Combined Dimensions")
                                st.markdown(f"**Analyzing combinations of: {', '.join(available_dimensions)}**")
                                
                                # Get biggest declines (most negative Net_Sale_Diff)
                                biggest_declines = combined_decline_df[combined_decline_df['Net_Sale_Diff'] < 0].sort_values('Net_Sale_Diff').head(15)
                                
                                if len(biggest_declines) > 0:
                                    # Format for display
                                    decline_display = biggest_declines[['Combined_Dimensions', 'Previous_Net_Sale', 'Latest_Net_Sale', 
                                                                     'Net_Sale_Diff', 'Net_Sale_Pct_Change', 'Discount_Diff', 'Discount_Pct_Diff']].copy()
                                    
                                    # Rename columns for clarity
                                    decline_display.columns = [
                                        f'{" + ".join(available_dimensions)} Combination',
                                        f'Previous Day Net Sales ({previous_date})',
                                        f'Latest Day Net Sales ({latest_date})',
                                        'Net Sales Change',
                                        'Net Sales % Change',
                                        'Discount Change',
                                        'Discount % Change'
                                    ]
                                    
                                    # Round values
                                    for col in decline_display.columns:
                                        if col != f'{" + ".join(available_dimensions)} Combination':
                                            if 'Change' in col and '%' in col:
                                                decline_display[col] = decline_display[col].round(1)
                                            else:
                                                decline_display[col] = decline_display[col].round(2)
                                    
                                    # Apply styling
                                    styled_decline = decline_display.style.format({
                                        col: '{:.2f}' if '% Change' not in col else '{:.1f}%'
                                        for col in decline_display.columns if col != f'{" + ".join(available_dimensions)} Combination'
                                    })
                                    
                                    # Apply conditional formatting
                                    styled_decline = styled_decline.applymap(style_differences, subset=['Net Sales Change'])
                                    styled_decline = styled_decline.applymap(style_discount_differences, subset=['Discount Change', 'Discount % Change'])
                                    
                                    st.dataframe(styled_decline, use_container_width=True)
                                    
                                    # Download option
                                    csv_declines = decline_display.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Combined Declines",
                                        data=csv_declines,
                                        file_name=f"combined_declines_{start_date}_{end_date}{time_suffix}.csv",
                                        mime="text/csv",
                                        key="download_combined_declines"
                                    )
                                else:
                                    st.info("🎉 No declines found - all combinations improved or stayed the same!")
                            
                            with decline_tab2:
                                st.markdown("### 📈 Biggest Improvements - Combined Dimensions")
                                st.markdown(f"**Analyzing combinations of: {', '.join(available_dimensions)}**")
                                
                                # Get biggest improvements (most positive Net_Sale_Diff)
                                biggest_improvements = combined_decline_df[combined_decline_df['Net_Sale_Diff'] > 0].sort_values('Net_Sale_Diff', ascending=False).head(15)
                                
                                if len(biggest_improvements) > 0:
                                    # Format for display
                                    improvement_display = biggest_improvements[['Combined_Dimensions', 'Previous_Net_Sale', 'Latest_Net_Sale', 
                                                                             'Net_Sale_Diff', 'Net_Sale_Pct_Change', 'Discount_Diff', 'Discount_Pct_Diff']].copy()
                                    
                                    # Rename columns for clarity
                                    improvement_display.columns = [
                                        f'{" + ".join(available_dimensions)} Combination',
                                        f'Previous Day Net Sales ({previous_date})',
                                        f'Latest Day Net Sales ({latest_date})',
                                        'Net Sales Change',
                                        'Net Sales % Change',
                                        'Discount Change',
                                        'Discount % Change'
                                    ]
                                    
                                    # Round values
                                    for col in improvement_display.columns:
                                        if col != f'{" + ".join(available_dimensions)} Combination':
                                            if 'Change' in col and '%' in col:
                                                improvement_display[col] = improvement_display[col].round(1)
                                            else:
                                                improvement_display[col] = improvement_display[col].round(2)
                                    
                                    # Apply styling
                                    styled_improvement = improvement_display.style.format({
                                        col: '{:.2f}' if '% Change' not in col else '{:.1f}%'
                                        for col in improvement_display.columns if col != f'{" + ".join(available_dimensions)} Combination'
                                    })
                                    
                                    # Apply conditional formatting
                                    styled_improvement = styled_improvement.applymap(style_differences, subset=['Net Sales Change'])
                                    styled_improvement = styled_improvement.applymap(style_discount_differences, subset=['Discount Change', 'Discount % Change'])
                                    
                                    st.dataframe(styled_improvement, use_container_width=True)
                                    
                                    # Download option
                                    csv_improvements = improvement_display.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Combined Improvements",
                                        data=csv_improvements,
                                        file_name=f"combined_improvements_{start_date}_{end_date}{time_suffix}.csv",
                                        mime="text/csv",
                                        key="download_combined_improvements"
                                    )
                                else:
                                    st.info("📊 No improvements found in the selected period")
                            
                            with decline_tab3:
                                st.markdown("### 🔄 All Changes - Combined Dimensions")
                                st.markdown(f"**Analyzing combinations of: {', '.join(available_dimensions)}**")
                                
                                # Show all data sorted by absolute change (biggest changes first)
                                combined_decline_df['Abs_Net_Sale_Diff'] = combined_decline_df['Net_Sale_Diff'].abs()
                                all_changes = combined_decline_df.sort_values('Abs_Net_Sale_Diff', ascending=False).head(20)
                                
                                # Format for display
                                all_display = all_changes[['Combined_Dimensions', 'Previous_Net_Sale', 'Latest_Net_Sale', 
                                                         'Net_Sale_Diff', 'Net_Sale_Pct_Change', 'Discount_Diff', 'Discount_Pct_Diff']].copy()
                                
                                # Rename columns for clarity
                                all_display.columns = [
                                    f'{" + ".join(available_dimensions)} Combination',
                                    f'Previous Day Net Sales ({previous_date})',
                                    f'Latest Day Net Sales ({latest_date})',
                                    'Net Sales Change',
                                    'Net Sales % Change',
                                    'Discount Change',
                                    'Discount % Change'
                                ]
                                
                                # Round values
                                for col in all_display.columns:
                                    if col != f'{" + ".join(available_dimensions)} Combination':
                                        if 'Change' in col and '%' in col:
                                            all_display[col] = all_display[col].round(1)
                                        else:
                                            all_display[col] = all_display[col].round(2)
                                
                                # Apply styling
                                styled_all = all_display.style.format({
                                    col: '{:.2f}' if '% Change' not in col else '{:.1f}%'
                                    for col in all_display.columns if col != f'{" + ".join(available_dimensions)} Combination'
                                })
                                
                                # Apply conditional formatting
                                styled_all = styled_all.applymap(style_differences, subset=['Net Sales Change'])
                                styled_all = styled_all.applymap(style_discount_differences, subset=['Discount Change', 'Discount % Change'])
                                
                                st.dataframe(styled_all, use_container_width=True)
                                
                                # Download option
                                csv_all_changes = all_display.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download All Combined Changes",
                                    data=csv_all_changes,
                                    file_name=f"all_combined_changes_{start_date}_{end_date}{time_suffix}.csv",
                                    mime="text/csv",
                                    key="download_all_combined_changes"
                                )
                            
                            # Summary metrics for combined decline analysis
                            st.markdown("### 📊 Combined Analysis Summary")
                            
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            with summary_col1:
                                total_declines = len(combined_decline_df[combined_decline_df['Net_Sale_Diff'] < 0])
                                st.metric(
                                    label="📉 Combinations with Declines",
                                    value=total_declines,
                                    delta=f"Out of {len(combined_decline_df)} total combinations"
                                )
                            
                            with summary_col2:
                                total_improvements = len(combined_decline_df[combined_decline_df['Net_Sale_Diff'] > 0])
                                st.metric(
                                    label="📈 Combinations with Improvements",
                                    value=total_improvements,
                                    delta=f"Out of {len(combined_decline_df)} total combinations"
                                )
                            
                            with summary_col3:
                                if total_declines > 0:
                                    worst_decline = combined_decline_df.loc[combined_decline_df['Net_Sale_Diff'].idxmin()]
                                    st.metric(
                                        label="💔 Worst Decline",
                                        value=worst_decline['Combined_Dimensions'][:50] + "...",  # Truncate for display
                                        delta=f"{worst_decline['Net_Sale_Diff']:,.0f}"
                                    )
                                else:
                                    st.metric(label="💔 Worst Decline", value="None", delta="All positive!")
                            
                            with summary_col4:
                                if total_improvements > 0:
                                    best_improvement = combined_decline_df.loc[combined_decline_df['Net_Sale_Diff'].idxmax()]
                                    st.metric(
                                        label="🏆 Best Improvement",
                                        value=best_improvement['Combined_Dimensions'][:50] + "...",  # Truncate for display
                                        delta=f"+{best_improvement['Net_Sale_Diff']:,.0f}"
                                    )
                                else:
                                    st.metric(label="🏆 Best Improvement", value="None", delta="No improvements")
                        
                        else:
                            st.warning("⚠️ No data available for combined dimension analysis")
                    
                    else:
                        st.warning("⚠️ No suitable dimensions (Brand, Location, Channel) found for combined analysis")
                
                except Exception as e:
                    st.error(f"❌ Error creating combined dimension analysis: {str(e)}")
            
            # Summary insights for combined table
            st.subheader("💡 Combined Analysis Insights")
            
            try:
                # Extract data for analysis
                net_sale_cols = [col for col in combined_pivot.columns if col[1] == 'Net Sale']
                discount_cols = [col for col in combined_pivot.columns if col[1] == 'Discount']
                
                if net_sale_cols and len(combined_pivot) > 1:
                    # Channel performance analysis (excluding totals row)
                    channel_data = combined_pivot.iloc[:-1]
                    
                    # Top performing channels by total net sales
                    total_net_sales = channel_data[net_sale_cols].sum(axis=1).sort_values(ascending=False)
                    top_channel = total_net_sales.index[0] if len(total_net_sales) > 0 else "N/A"
                    top_channel_sales = total_net_sales.iloc[0] if len(total_net_sales) > 0 else 0
                    
                    # Channel with highest average discount percentage
                    if discount_cols:
                        avg_discount_by_channel = channel_data[discount_cols].sum(axis=1) / channel_data[net_sale_cols].sum(axis=1) * 100
                        highest_discount_channel = avg_discount_by_channel.idxmax()
                        highest_discount_pct = avg_discount_by_channel.max()
                    else:
                        highest_discount_channel = "N/A"
                        highest_discount_pct = 0
                    
                    # Date performance analysis
                    date_totals = {}
                    for col in net_sale_cols:
                        date = col[0]
                        date_totals[date] = combined_pivot[col].iloc[-1]  # Get total row value
                    
                    if date_totals:
                        best_date = max(date_totals, key=date_totals.get)
                        best_date_sales = date_totals[best_date]
                    else:
                        best_date = "N/A"
                        best_date_sales = 0
                    
                    # Display insights
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric(
                            label="🏆 Top Performing Channel",
                            value=top_channel,
                            delta=f"{top_channel_sales:,.0f} total sales"
                        )
                    
                    with insight_col2:
                        st.metric(
                            label="� Highest Discount Channel",
                            value=highest_discount_channel,
                            delta=f"{highest_discount_pct:.1f}% avg discount"
                        )
                    
                    with insight_col3:
                        st.metric(
                            label="📅 Best Sales Date",
                            value=best_date,
                            delta=f"{best_date_sales:,.0f} total sales"
                        )
                    
                    # Additional insights
                    st.write("� **Key Observations:**")
                    
                    observations = []
                    
                    if len(total_net_sales) >= 2:
                        second_best = total_net_sales.index[1]
                        performance_gap = total_net_sales.iloc[0] - total_net_sales.iloc[1]
                        observations.append(f"• **{top_channel}** outperforms **{second_best}** by {performance_gap:,.0f} in total sales")
                    
                    if highest_discount_pct > 10:
                        observations.append(f"• **{highest_discount_channel}** has the highest discount rate at {highest_discount_pct:.1f}%")
                    
                    if len(date_totals) >= 2:
                        dates_sorted = sorted(date_totals.items(), key=lambda x: x[1], reverse=True)
                        if len(dates_sorted) >= 2:
                            trend_direction = "increasing" if dates_sorted[0][1] > dates_sorted[-1][1] else "decreasing"
                            observations.append(f"• Sales trend appears to be **{trend_direction}** across the selected period")
                    
                    for obs in observations:
                        st.write(obs)
                    
                    # Day-to-day difference insights (if we have exactly 2 dates)
                    if len(dates) == 2:
                        st.write("📊 **Day-to-Day Changes:**")
                        
                        # Get difference columns
                        diff_cols = [col for col in combined_pivot.columns if 'Δ' in col[0]]
                        net_sale_diff_col = next((col for col in diff_cols if 'Net Sale Δ' in col[1]), None)
                        discount_diff_col = next((col for col in diff_cols if 'Discount Δ' in col[1]), None)
                        discount_pct_diff_col = next((col for col in diff_cols if 'Discount % Δ' in col[1]), None)
                        
                        if net_sale_diff_col:
                            # Channel with biggest improvement/decline
                            channel_differences = channel_data[net_sale_diff_col].sort_values(ascending=False)
                            
                            if len(channel_differences) > 0:
                                best_improvement = channel_differences.index[0]
                                best_improvement_value = channel_differences.iloc[0]
                                
                                worst_decline = channel_differences.index[-1]
                                worst_decline_value = channel_differences.iloc[-1]
                                
                                st.write(f"• **Best Improvement**: {best_improvement} (+{best_improvement_value:,.0f} net sales)")
                                st.write(f"• **Biggest Decline**: {worst_decline} ({worst_decline_value:,.0f} net sales)")
                                
                                # Overall trend
                                total_diff = combined_pivot[net_sale_diff_col].iloc[-1]  # Total row
                                trend_emoji = "📈" if total_diff > 0 else "📉" if total_diff < 0 else "➡️"
                                st.write(f"• **Overall Trend**: {trend_emoji} {total_diff:+,.0f} total net sales change")
                    
                else:
                    st.write("💡 Unable to generate insights - insufficient data in the combined table.")
            
            except Exception as e:
                st.write("💡 Unable to generate insights for this combined data.")
                st.write(f"Error details: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error creating combined pivot table: {str(e)}")
            st.write("Please check your data and try again.")
    
    elif use_quick_setup and 'Channel' not in df_filtered.columns:
        st.warning("⚠️ **Channel column not found** in the data. Quick Setup requires a 'Channel' column.")
        st.info("💡 Use the Manual Configuration below to create pivot tables with available columns.")
    
    else:
        # Manual configuration (existing code)
        st.subheader("🔧 Manual Pivot Table Configuration")
        
        # Pivot table configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📋 Rows (Index)**")
            rows = st.multiselect(
                "Select row dimensions:",
                options=available_categorical,
                default=['Channel'] if 'Channel' in available_categorical else (['DayOfWeek'] if 'DayOfWeek' in available_categorical else available_categorical[:1]),
                help="Choose categorical columns to group by in rows"
            )
        
        with col2:
            st.write("**📊 Columns**")
            columns = st.multiselect(
                "Select column dimensions:",
                options=available_categorical,
                default=['Date'] if 'Date' in available_categorical else (['Month'] if 'Month' in available_categorical else []),
                help="Choose categorical columns to group by in columns (optional)"
            )
        
        with col3:
            st.write("**📈 Values**")
            values = st.selectbox(
                "Select value to aggregate:",
                options=available_numeric,
                index=0 if available_numeric else None,
                help="Choose numeric column to aggregate"
            )
            
            aggregation = st.selectbox(
                "Aggregation function:",
                options=['sum', 'mean', 'count', 'median', 'std', 'min', 'max'],
                index=0,
                help="Choose how to aggregate the values"
            )
        
        if not rows or not values:
            st.warning("⚠️ Please select at least one row dimension and one value column.")
            return
        
        # Create manual pivot table (existing pivot table code)
        try:
            # Create pivot table
            if columns:
                pivot_table = df_filtered.pivot_table(
                    values=values,
                    index=rows,
                    columns=columns,
                    aggfunc=aggregation,
                    fill_value=0,
                    margins=True,
                    margins_name="Total"
                )
            else:
                # Group by without pivot if no columns selected
                pivot_table = df_filtered.groupby(rows)[values].agg(aggregation).reset_index()
                pivot_table = pivot_table.round(2)
            
            # Display configuration summary
            st.info(f"📊 **Pivot Table**: {' × '.join(rows)} {'× ' + ' × '.join(columns) if columns else ''} | **Values**: {values} ({aggregation})")
            
            # Display the pivot table
            st.subheader("📊 Pivot Table Results")
            st.dataframe(pivot_table.round(2), use_container_width=True)
            
            # Download option
            csv_data = pivot_table.to_csv()
            st.download_button(
                label="📥 Download Pivot Table as CSV",
                data=csv_data,
                file_name=f"pivot_table_{values}_{aggregation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Existing visualization and insights code...
            # [Rest of the manual pivot table code continues here]
            
        except Exception as e:
            st.error(f"❌ Error creating pivot table: {str(e)}")
            st.write("Please try a different combination of dimensions and values.")

# Utility function to suppress WebSocket errors
def suppress_websocket_errors():
    """Context manager to suppress WebSocket-related errors"""
    import contextlib
    import sys
    from io import StringIO
    
    @contextlib.contextmanager
    def suppress_stderr():
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    
    return suppress_stderr()

def main():
    st.markdown('<div class="main-header">🚀 Sales Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Data loading and filtering notice
    st.info("🚀 **Smart Loading**: Dashboard starts with 1 week of data for quick access, then automatically loads another week in background.")
    st.info("ℹ️ **Data Filtering**: Orders with 100% discounts (promotional/test orders) are automatically excluded from all analyses to ensure accurate business insights.")
    
    # Sidebar for data source selection
    st.sidebar.header("📊 Dashboard Configuration")
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["BigQuery", "MySQL"],  # BigQuery first (recommended)
        help="Choose your preferred data source (BigQuery recommended for better performance)"
    )
    
    # Add performance note
    if data_source == "BigQuery":
        st.sidebar.success("✅ BigQuery: Optimized for analytics")
    else:
        st.sidebar.info("💡 Consider BigQuery for better performance")
    
    # Smart Data Loading Strategy
    st.sidebar.subheader("📊 Data Loading")
    
    # Initialize session state for data management
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.background_loaded = False
        st.session_state.current_data_range = 7  # Start with 1 week for fast loading
        st.session_state.df = pd.DataFrame()
    
    # Load initial data (1 week for fast start)
    if not st.session_state.data_loaded:
        with st.sidebar:
            with st.spinner("� Quick loading (1 week)..."):
                if data_source == "MySQL":
                    st.session_state.df = load_data_from_mysql(days_back=7)
                else:
                    st.session_state.df = load_data_from_bigquery(days_back=7)
                
                st.session_state.data_loaded = True
                st.session_state.current_data_range = 7
                
        if not st.session_state.df.empty:
            st.sidebar.success(f"✅ Loaded {len(st.session_state.df):,} records (1 week) - Loading more in background...")
        else:
            st.sidebar.error("❌ No data loaded")
    
    # Background loading of additional week
    if st.session_state.data_loaded and not st.session_state.background_loaded and st.session_state.current_data_range == 7:
        try:
            with st.sidebar:
                with st.spinner("📊 Loading additional week in background..."):
                    if data_source == "MySQL":
                        df_extended = load_data_from_mysql(days_back=14)
                    else:
                        df_extended = load_data_from_bigquery(days_back=14)
                    
                    if not df_extended.empty and len(df_extended) > len(st.session_state.df):
                        st.session_state.df = df_extended
                        st.session_state.current_data_range = 14
                        st.session_state.background_loaded = True
                        st.sidebar.success(f"✅ Background load complete! Now showing {len(df_extended):,} records (2 weeks)")
                        st.rerun()
                    else:
                        st.session_state.background_loaded = True  # Mark as completed even if no additional data
        except Exception as e:
            st.sidebar.warning(f"⚠️ Background loading failed: {str(e)}")
            st.session_state.background_loaded = True  # Mark as completed to avoid retry loops
    
    df = st.session_state.df.copy()
    
    if df.empty:
        st.error("No data available. Please check your database connection.")
        return
    
    # Check data date range and provide expansion options
    min_date_available = df['Date'].min()
    max_date_available = df['Date'].max()
    days_loaded = (max_date_available - min_date_available).days + 1
    
    # Show current data range info with background loading status
    if st.session_state.background_loaded:
        st.sidebar.info(f"📅 **Current Data**: {min_date_available} to {max_date_available} ({days_loaded} days) ✅")
    else:
        st.sidebar.info(f"📅 **Current Data**: {min_date_available} to {max_date_available} ({days_loaded} days) 🔄")
    
    # Option to load more historical data
    if st.session_state.current_data_range < 365:  # If not loading full year yet
        
        # Add reset option if currently loading more data than 7 days
        current_range = st.session_state.current_data_range
        
        if current_range != 7:
            if st.sidebar.button("🔄 Reset to 1 Week", help="Reset to quick 1-week view"):
                with st.spinner("🔄 Loading 1 week of data..."):
                    if data_source == "MySQL":
                        df_reset = load_data_from_mysql(days_back=7)
                    else:
                        df_reset = load_data_from_bigquery(days_back=7)
                    
                    if not df_reset.empty:
                        st.session_state.df = df_reset
                        st.session_state.current_data_range = 7
                        st.session_state.background_loaded = False  # Reset background loading flag
                        st.sidebar.success(f"✅ Reset to {len(df_reset):,} records (1 week)")
                        st.rerun()
        
        # Data expansion options
        st.sidebar.subheader("📊 Load More Data")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📅 2 Weeks", help="Load complete data for last 14 days"):
                with st.spinner("🔄 Loading 2 weeks of complete data..."):
                    if data_source == "MySQL":
                        df_extended = load_data_from_mysql(days_back=14)
                    else:
                        df_extended = load_data_from_bigquery(days_back=14)
                    
                    if not df_extended.empty:
                        st.session_state.df = df_extended
                        st.session_state.current_data_range = 14
                        st.session_state.background_loaded = True  # Mark as completed
                        st.sidebar.success(f"✅ Loaded {len(df_extended):,} records (2 weeks)")
                        st.rerun()
        
        with col2:
            if st.button("� 1 Month", help="Load complete data for last 30 days"):
                with st.spinner("🔄 Loading 1 month of complete data..."):
                    if data_source == "MySQL":
                        df_extended = load_data_from_mysql(days_back=30)
                    else:
                        df_extended = load_data_from_bigquery(days_back=30)
                    
                    if not df_extended.empty:
                        st.session_state.df = df_extended
                        st.session_state.current_data_range = 30
                        st.sidebar.success(f"✅ Loaded {len(df_extended):,} records (1 month)")
                        st.rerun()
        
        # Additional options in a new row
        col3, col4 = st.sidebar.columns(2)
        
        with col3:
            if st.button("📈 3 Months", help="Load complete data for last 90 days"):
                with st.spinner("🔄 Loading 3 months of complete data..."):
                    if data_source == "MySQL":
                        df_extended = load_data_from_mysql(days_back=90)
                    else:
                        df_extended = load_data_from_bigquery(days_back=90)
                    
                    if not df_extended.empty:
                        st.session_state.df = df_extended
                        st.session_state.current_data_range = 90
                        st.sidebar.success(f"✅ Loaded {len(df_extended):,} records (3 months)")
                        st.rerun()
        
        with col3:
            if st.button("� 6 Months", help="Load complete data for last 180 days"):
                with st.spinner("🔄 Loading 6 months of complete data..."):
                    if data_source == "MySQL":
                        df_extended = load_data_from_mysql(days_back=180)
                    else:
                        df_extended = load_data_from_bigquery(days_back=180)
                    
                    if not df_extended.empty:
                        st.session_state.df = df_extended
                        st.session_state.current_data_range = 180
                        st.sidebar.success(f"✅ Loaded {len(df_extended):,} records (6 months)")
                        st.rerun()
        
        with col4:
            if st.button("� 1 Year", help="Load complete data for full year"):
                with st.spinner("🔄 Loading 1 year of complete data..."):
                    if data_source == "MySQL":
                        df_extended = load_data_from_mysql(days_back=365)
                    else:
                        df_extended = load_data_from_bigquery(days_back=365)
                    
                    if not df_extended.empty:
                        st.session_state.df = df_extended
                        st.session_state.current_data_range = 365
                        st.sidebar.success(f"✅ Loaded {len(df_extended):,} records (1 year)")
                        st.rerun()
    
    # Option to load ALL historical data
    if st.sidebar.button("🗂️ Load All Historical Data", help="Load complete historical dataset (may take longer)"):
        with st.spinner("🔄 Loading all historical data..."):
            if data_source == "MySQL":
                df_all = load_data_from_mysql(days_back=None)
            else:
                df_all = load_data_from_bigquery(days_back=None)
            
            if not df_all.empty:
                st.session_state.df = df_all
                st.session_state.current_data_range = None
                st.sidebar.success(f"✅ Loaded {len(df_all):,} records (all historical data)")
                st.rerun()
    
    # Auto-expand data if user selects dates outside current range
    df = st.session_state.df.copy()
    
    # Filter out orders with 100% discounts (promotional/test orders)
    initial_count = len(df)
    if 'Discount_Percentage' in df.columns:
        df = df[df['Discount_Percentage'] < 100.0]
        filtered_count = len(df)
        excluded_count = initial_count - filtered_count
        
        if excluded_count > 0:
            st.sidebar.info(f"🚫 Excluded {excluded_count:,} orders with 100% discount")
    
    # Date filter with smart expansion
    st.sidebar.subheader("📅 Date Filter")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Show available date range
    st.sidebar.write(f"**Available Range**: {min_date} to {max_date}")
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        help="Select dates within the currently loaded range. Use 'Load More Data' buttons above to expand the range."
    )
    
    # Smart data expansion: Check if user selected dates outside current range
    if len(date_range) == 2:
        selected_start, selected_end = date_range
        days_requested = (selected_end - selected_start).days + 1
        
        # If user selected a range outside current data, suggest loading more
        if selected_start < min_date or selected_end > max_date:
            st.sidebar.warning("⚠️ Selected dates are outside current data range. Use 'Load More Data' buttons above.")
        
        # Apply date filter
        df = df[(df['Date'] >= selected_start) & (df['Date'] <= selected_end)]
        
        # Show filtered range info
        if len(df) > 0:
            actual_start = df['Date'].min()
            actual_end = df['Date'].max()
            st.sidebar.info(f"📊 **Filtered**: {actual_start} to {actual_end} ({len(df):,} records)")
        else:
            st.sidebar.warning("⚠️ No data found in selected date range")
    
    # Additional filters
    if 'Channel' in df.columns:
        channels = st.sidebar.multiselect(
            "Select Channels",
            options=df['Channel'].unique(),
            default=df['Channel'].unique()
        )
        df = df[df['Channel'].isin(channels)]
    
    if 'Brand' in df.columns:
        brands = st.sidebar.multiselect(
            "Select Brands",
            options=df['Brand'].unique(),
            default=df['Brand'].unique()
        )
        df = df[df['Brand'].isin(brands)]
    
    # Calculate and display key metrics
    metrics = calculate_metrics(df)
    
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"{metrics.get('total_revenue', 0):,.2f}",
            delta=f"{metrics.get('total_orders', 0)} orders"
        )
    
    with col2:
        st.metric(
            label="🛒 Average Order Value",
            value=f"{metrics.get('avg_order_value', 0):,.2f}",
            delta=f"{metrics.get('total_customers', 0)} customers"
        )
    
    with col3:
        st.metric(
            label="🎯 Discount Rate",
            value=f"{metrics.get('discount_rate', 0):.1f}%",
            delta=f"{metrics.get('total_discount', 0):,.2f} total"
        )
    
    with col4:
        st.metric(
            label="💡 Tips Collected",
            value=f"{metrics.get('total_tips', 0):,.2f}",
            delta=f"{metrics.get('avg_delivery_fee', 0):.2f} avg delivery"
        )
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Sales Trends", "👥 Customer Behavior", "🎫 Discount Performance", 
        "🔮 Sales Predictions", "⚖️ Comparative Analysis", "🧮 Advanced Metrics", "To Sir Shady", "📋 Data Overview"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">📊 Sales Trends Analysis</div>', unsafe_allow_html=True)
        trends_fig = create_sales_trends_chart(df)
        st.plotly_chart(trends_fig, use_container_width=True)
        
        # Additional trend insights
        col1, col2 = st.columns(2)
        with col1:
            # Hourly sales pattern
            hourly_sales = df.groupby('Hour')['net_sale'].sum()
            fig_hourly = px.bar(x=hourly_sales.index, y=hourly_sales.values,
                               title="Sales by Hour of Day")
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Day of week pattern
            dow_sales = df.groupby('DayOfWeek')['net_sale'].sum()
            fig_dow = px.bar(x=dow_sales.index, y=dow_sales.values,
                            title="Sales by Day of Week")
            st.plotly_chart(fig_dow, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">👥 Customer Behavior Analysis</div>', unsafe_allow_html=True)
        customer_fig, customer_stats = create_customer_behavior_analysis(df)
        st.plotly_chart(customer_fig, use_container_width=True)
        
        st.subheader("🏆 Top Customers")
        top_customers = customer_stats.nlargest(10, 'Total_Spent')
        st.dataframe(top_customers, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">🎫 Discount Performance Analysis</div>', unsafe_allow_html=True)
        discount_fig = create_discount_performance_analysis(df)
        st.plotly_chart(discount_fig, use_container_width=True)
        
        # Discount insights
        if not df[df['Discount'] > 0].empty:
            discount_insights = df[df['Discount'] > 0].groupby('Date').agg({
                'Discount': ['sum', 'mean', 'count'],
                'net_sale': 'sum'
            }).round(2)
            st.subheader("📊 Daily Discount Summary")
            st.dataframe(discount_insights.tail(10), use_container_width=True)
    
    with tab4:
        st.markdown('<div class="section-header">🔮 Sales Predictions</div>', unsafe_allow_html=True)
        pred_fig, model_metrics = create_sales_predictions(df)
        
        if model_metrics:
            st.plotly_chart(pred_fig, use_container_width=True)
            
            st.subheader("🎯 Model Performance")
            metrics_df = pd.DataFrame(model_metrics).round(3)
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("Unable to generate predictions. Please ensure sufficient data is available.")
    
    with tab5:
        st.markdown('<div class="section-header">⚖️ Comparative Analysis</div>', unsafe_allow_html=True)
        comparison_fig = create_comparison_analysis(df)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Additional comparison tables
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Channel' in df.columns:
                st.subheader("📱 Channel Performance")
                channel_metrics = df.groupby('Channel').agg({
                    'net_sale': ['sum', 'mean'],
                    'OrderID': 'count',
                    'Discount': 'sum'
                }).round(2)
                st.dataframe(channel_metrics, use_container_width=True)
        
        with col2:
            if 'PaymentMethod' in df.columns:
                st.subheader("💳 Payment Method Analysis")
                payment_metrics = df.groupby('PaymentMethod').agg({
                    'net_sale': ['sum', 'mean'],
                    'OrderID': 'count'
                }).round(2)
                st.dataframe(payment_metrics, use_container_width=True)
    
    with tab6:
        st.markdown('<div class="section-header">🧮 Advanced Metrics & Calculated Columns</div>', unsafe_allow_html=True)
        
        # ✅ NET_SALE COLUMN ANALYSIS
        st.markdown('<div class="section-header">🎯 Custom Net Sale Analysis</div>', unsafe_allow_html=True)
        
        if 'net_sale' in df.columns:
            # Net Sale Overview Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_net_sale = df['net_sale'].sum()
                st.metric(
                    label="📊 Total Net Sale",
                    value=f"{total_net_sale:,.2f}",
                    delta=f"Calculated Column"
                )
            
            with col2:
                avg_net_sale = df['net_sale'].mean()
                st.metric(
                    label="📈 Avg Net Sale",
                    value=f"{avg_net_sale:,.2f}",
                    delta=f"Per Order"
                )
            
            with col3:
                net_sale_vs_recorded = (df['net_sale'] - df['NetSales']).mean()
                st.metric(
                    label="🔄 Difference vs Recorded",
                    value=f"{net_sale_vs_recorded:,.2f}",
                    delta=f"Calculated - Recorded"
                )
            
            with col4:
                special_cases = (df['GrossPrice'] == df['Discount']).sum()
                st.metric(
                    label="⚠️ Special Cases",
                    value=f"{special_cases}",
                    delta=f"GrossPrice = Discount"
                )
            
            # Net Sale Data Table - ONLY NET_SALE COLUMN DATA
            st.subheader("📋 Net Sale Column Data (Custom Calculated)")
            
            # Create a focused dataframe with key columns for context and the net_sale column
            net_sale_df = df[['OrderID', 'CustomerName', 'GrossPrice', 'Discount', 'NetSales', 'net_sale']].copy()
            
            # Add helper columns to understand the calculation
            net_sale_df['Formula_Used'] = np.where(
                df['GrossPrice'] == df['Discount'],
                '(GrossPrice/1.05) - (Discount/1.05)',
                'GrossPrice/1.05'
            )
            
            net_sale_df['Difference_from_Recorded'] = net_sale_df['net_sale'] - net_sale_df['NetSales']
            net_sale_df['Special_Case'] = net_sale_df['GrossPrice'] == net_sale_df['Discount']
            
            # Sort by net_sale value (highest first)
            net_sale_df = net_sale_df.sort_values('net_sale', ascending=False)
            
            st.dataframe(net_sale_df, use_container_width=True)
            
            # Download NET_SALE specific data
            net_sale_csv = net_sale_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Net Sale Data CSV",
                data=net_sale_csv,
                file_name=f"net_sale_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Net Sale Analysis Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Net Sale Distribution
                fig_dist = px.histogram(df, x='net_sale', nbins=30, 
                                      title="Net Sale Distribution")
                fig_dist.update_layout(
                    xaxis_title="Net Sale Amount",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Net Sale vs Recorded NetSales Comparison
                fig_comparison = px.scatter(df, x='NetSales', y='net_sale',
                                          title="Net Sale vs Recorded NetSales",
                                          hover_data=['OrderID', 'CustomerName'])
                # Add diagonal line for perfect match
                min_val = min(df['NetSales'].min(), df['net_sale'].min())
                max_val = max(df['NetSales'].max(), df['net_sale'].max())
                fig_comparison.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash"),
                )
                fig_comparison.update_layout(
                    xaxis_title="Recorded NetSales",
                    yaxis_title="Calculated Net Sale"
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Summary Statistics for Net Sale
            st.subheader("📊 Net Sale Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Net Sale Summary:**")
                st.write(f"• Min Net Sale: {df['net_sale'].min():,.2f}")
                st.write(f"• Max Net Sale: {df['net_sale'].max():,.2f}")
                st.write(f"• Median Net Sale: {df['net_sale'].median():,.2f}")
                st.write(f"• Standard Deviation: {df['net_sale'].std():,.2f}")
            
            with col2:
                st.write("**🔍 Calculation Analysis:**")
                normal_cases = (df['GrossPrice'] != df['Discount']).sum()
                st.write(f"• Normal Cases (GrossPrice/1.05): {normal_cases}")
                st.write(f"• Special Cases (Formula): {special_cases}")
                
                if len(df) > 0:
                    normal_pct = (normal_cases / len(df)) * 100
                    special_pct = (special_cases / len(df)) * 100
                    st.write(f"• Normal Cases: {normal_pct:.1f}%")
                    st.write(f"• Special Cases: {special_pct:.1f}%")
        else:
            st.error("❌ Net Sale column not found. Please check the calculation.")
        
        st.markdown("---")  # Separator
        
        # Performance Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_profit_margin = df['Profit_Margin'].mean()
            st.metric(
                label="📈 Avg Profit Margin",
                value=f"{avg_profit_margin:.1f}%",
                delta=f"{df['Profit_Margin'].std():.1f}% volatility"
            )
        
        with col2:
            high_value_orders = (df['High_Value_Order'].sum() / len(df) * 100)
            st.metric(
                label="⭐ High Value Orders",
                value=f"{high_value_orders:.1f}%",
                delta=f"{df['High_Value_Order'].sum()} orders"
            )
        
        with col3:
            avg_tip_percent = df['Tip_Percentage'].mean()
            st.metric(
                label="💰 Avg Tip Rate",
                value=f"{avg_tip_percent:.1f}%",
                delta=f"{df['Tips'].sum():,.0f} total"
            )
        
        with col4:
            weekend_revenue = df[df['Is_Weekend']]['net_sale'].sum()
            weekend_percent = (weekend_revenue / df['net_sale'].sum() * 100)
            st.metric(
                label="🎯 Weekend Revenue",
                value=f"{weekend_percent:.1f}%",
                delta=f"{weekend_revenue:,.0f}"
            )
        
        # Advanced Analytics Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Order Size Distribution
            if 'Order_Size_Category' in df.columns:
                size_dist = df['Order_Size_Category'].value_counts()
                fig_size = px.pie(values=size_dist.values, names=size_dist.index,
                                title="Order Size Distribution")
                st.plotly_chart(fig_size, use_container_width=True)
        
        with col2:
            # Profit Margin by Time Period
            if 'Time_Period' in df.columns:
                margin_by_time = df.groupby('Time_Period')['Profit_Margin'].mean()
                fig_margin = px.bar(x=margin_by_time.index, y=margin_by_time.values,
                                  title="Profit Margin by Time Period")
                st.plotly_chart(fig_margin, use_container_width=True)
        
        # Customer Value Analysis
        if 'Customer_Value_Tier' in df.columns:
            st.subheader("🏆 Customer Value Tier Analysis")
            tier_analysis = df.groupby('Customer_Value_Tier').agg({
                'net_sale': ['sum', 'mean', 'count'],
                'Profit_Margin': 'mean',
                'Discount_Percentage': 'mean'
            }).round(2)
            st.dataframe(tier_analysis, use_container_width=True)
        
        # Advanced Metrics Table
        st.subheader("📊 Sample of Calculated Columns")
        calculated_cols = ['Order_Size_Category', 'Discount_Category', 'Profit_Margin', 
                          'Discount_Percentage', 'Time_Period', 'Is_Weekend', 'High_Value_Order']
        available_cols = [col for col in calculated_cols if col in df.columns]
        
        if available_cols:
            sample_data = df[['CustomerName', 'net_sale', 'Discount'] + available_cols].head(20)
            st.dataframe(sample_data, use_container_width=True)
        
        # Business Insights
        st.subheader("💡 Key Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Performance Metrics:**")
            if 'Order_Profitability' in df.columns:
                avg_profitability = df['Order_Profitability'].mean()
                st.write(f"• Average Order Profitability: {avg_profitability:.2f}")
            
            if 'Premium_Customer' in df.columns:
                premium_count = df['Premium_Customer'].sum()
                st.write(f"• Premium Customers: {premium_count}")
            
            if 'Discounted_Order' in df.columns:
                discount_rate = (df['Discounted_Order'].sum() / len(df) * 100)
                st.write(f"• Orders with Discounts: {discount_rate:.1f}%")
        
               
        with col2:
            st.write("**⏰ Time-based Insights:**")
            if 'Is_Weekend' in df.columns:
                weekend_orders = df['Is_Weekend'].sum()
                st.write(f"• Weekend Orders: {weekend_orders}")
            
            if 'Time_Period' in df.columns:
                peak_time = df.groupby('Time_Period')['net_sale'].sum().idxmax()
                st.write(f"• Peak Sales Period: {peak_time}")
            
            if 'Order_Size_Category' in df.columns:
                most_common_size = df['Order_Size_Category'].mode().iloc[0] if not df['Order_Size_Category'].mode().empty else 'N/A'
                st.write(f"• Most Common Order Size: {most_common_size}")

    with tab7:
        create_pivot_table_analysis(df)

    with tab8:
        st.markdown('<div class="section-header">📋 Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Information")
            st.write(f"**Total Records:** {len(df):,}")
            st.write(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Data Source:** {data_source}")
        
        with col2:
            st.subheader("📈 Data Quality")
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            quality_df = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing %': missing_pct.round(2)
            })
            st.dataframe(quality_df[quality_df['Missing Count'] > 0], use_container_width=True)
        
        st.subheader("🔍 Sample Data")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log the error but don't show WebSocket errors to users
        if 'WebSocketClosedError' not in str(e) and 'Stream is closed' not in str(e):
            st.error(f"An error occurred: {str(e)}")
        # For WebSocket errors, just log and continue
        pass
