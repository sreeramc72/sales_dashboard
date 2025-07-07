# 📊 Sales Analytics Dashboard

A comprehensive sales analytics dashboard with deep insights, customer behavior analysis, discount performance tracking, sales predictions, and comparative analysis.

## 🚀 Features

### 📈 Deep Sales Analysis
- **Sales Trends**: Daily, hourly, and seasonal sales patterns
- **Revenue Tracking**: Real-time revenue monitoring and growth analysis
- **Order Analytics**: Order volume, frequency, and distribution analysis

### 👥 Customer Behavior Analysis
- **Customer Segmentation**: Automatic categorization (Low/Medium/High/VIP value customers)
- **Customer Lifetime Value**: CLV calculation and analysis
- **Repeat Customer Analysis**: Customer retention and loyalty metrics
- **Purchase Patterns**: Order frequency and behavior tracking

### 🎫 Discount Performance
- **Discount Effectiveness**: ROI analysis of discount campaigns
- **Discount Code Tracking**: Performance of specific promo codes
- **Discount Distribution**: Analysis of discount amounts and usage patterns
- **Impact Assessment**: How discounts affect order values and customer behavior

### 🔮 Sales Predictions
- **Machine Learning Models**: Linear Regression and Random Forest predictions
- **30-Day Forecasts**: Future sales predictions with confidence intervals
- **Trend Analysis**: Predictive insights for business planning
- **Model Performance**: Accuracy metrics and model comparison

### ⚖️ Comparative Analysis
- **Channel Performance**: Sales comparison across different channels
- **Location Analysis**: Geographic performance insights
- **Brand Comparison**: Multi-brand performance tracking
- **Payment Methods**: Payment preference analysis

### 📊 Interactive Dashboard
- **Real-time Data**: Live connection to MySQL/BigQuery
- **Dynamic Filtering**: Date range, channel, and brand filters
- **Export Capabilities**: Download data and reports
- **Responsive Design**: Works on desktop and mobile

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MySQL database with sales data
- BigQuery access (optional)
- Internet connection for package installation

### Quick Start

#### Option 1: Using PowerShell (Windows)
```powershell
# Navigate to the project directory
cd "C:\Users\sreer\OneDrive\Desktop\Dont delete\mysql"

# Run the PowerShell launcher
.\run_dashboard.ps1
```

#### Option 2: Using Python Launcher
```bash
# Navigate to the project directory
cd "C:\Users\sreer\OneDrive\Desktop\Dont delete\mysql"

# Run the Python launcher
python launch_dashboard.py
```

#### Option 3: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run sales_dashboard.py
```

### 🔧 Environment Setup

Ensure your `.env` file contains the following variables:
```env
# MySQL Configuration
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=order_level

# Grubtech Credentials (if using import.py)
GRUBTECH_USERNAME=your_username
GRUBTECH_PASSWORD=your_password
GRUBTECH_EXCEL_PASSWORD=your_excel_password
```

## 📊 Data Requirements

The dashboard expects a `sales_data` table with the following columns:

### Required Columns
- `ReceivedAt`: Order timestamp
- `OrderID`: Unique order identifier
- `CustomerName`: Customer name
- `NetSales`: Net sales amount
- `GrossPrice`: Gross price before discounts
- `Discount`: Discount amount

### Optional Columns (for enhanced analysis)
- `Channel`: Sales channel (online, mobile, etc.)
- `Brand`: Brand name
- `Location`: Store/location identifier
- `PaymentMethod`: Payment method used
- `DiscountCode`: Discount code applied
- `ItemPrice`: Item price
- `Delivery`: Delivery charges
- `Tips`: Tips amount
- `VAT`: Tax amount

## 🎯 Usage Guide

### 1. Launch the Dashboard
After running any of the setup options above, the dashboard will open in your browser at `http://localhost:8501`

### 2. Select Data Source
- Choose between MySQL or BigQuery from the sidebar
- The dashboard will automatically connect and load your data

### 3. Apply Filters
- **Date Range**: Filter data by specific date ranges
- **Channels**: Select specific sales channels
- **Brands**: Filter by brand if applicable

### 4. Explore Analytics

#### 📊 Sales Trends Tab
- View daily revenue and order trends
- Analyze hourly and day-of-week patterns
- Identify peak sales periods

#### 👥 Customer Behavior Tab
- Explore customer segmentation
- View top customers and their behavior
- Analyze customer lifetime value

#### 🎫 Discount Performance Tab
- Track discount effectiveness
- Analyze discount code performance
- Understand discount impact on sales

#### 🔮 Sales Predictions Tab
- View 30-day sales forecasts
- Compare different prediction models
- Use insights for business planning

#### ⚖️ Comparative Analysis Tab
- Compare performance across channels
- Analyze location-based sales
- Track payment method preferences

#### 📋 Data Overview Tab
- View data quality metrics
- Export data as CSV
- Inspect raw data

## 📈 Key Metrics Explained

### Revenue Metrics
- **Total Revenue**: Sum of all NetSales
- **Average Order Value**: Mean order value
- **Order Count**: Total number of orders

### Customer Metrics
- **Total Customers**: Unique customer count
- **Customer Segments**: Based on spending levels
- **Repeat Rate**: Percentage of returning customers

### Discount Metrics
- **Discount Rate**: Total discounts as % of gross sales
- **Discount Effectiveness**: Impact on order values
- **Code Performance**: ROI of specific discount codes

### Prediction Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Model accuracy (0-1, higher is better)

## 🔧 Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check Python installation
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Run with verbose logging
streamlit run sales_dashboard.py --logger.level=debug
streamlit run sales_dashboard.py --logger.level=debug
```

#### Database Connection Issues
1. Verify your `.env` file contains correct credentials
2. Test MySQL connection:
```python
import mysql.connector
conn = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='your_password',
    database='order_level'
)
print("Connection successful!")
```

#### Missing Data Columns
- Check that your `sales_data` table has the required columns
- Update the column mapping in the dashboard if needed

#### Performance Issues
- Filter data by date range for large datasets
- Consider indexing your database tables
- Use BigQuery for better performance with large datasets

## 🚀 Advanced Features

### Custom Metrics
You can modify the `calculate_metrics()` function to add custom KPIs specific to your business.

### Additional Visualizations
The dashboard is built with Plotly, making it easy to add new charts and visualizations.

### Data Export
- Export filtered data as CSV
- Save charts as images
- Generate PDF reports (can be added)

## 📊 Sample Dashboard Screenshots

The dashboard includes:
- 📈 Interactive sales trend charts
- 🥧 Customer segment pie charts
- 📊 Revenue comparison bar charts
- 🔮 Predictive analytics graphs
- 📋 Data tables with sorting and filtering

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data structure matches requirements
3. Ensure all dependencies are installed correctly

## 📝 Technical Details

### Built With
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning predictions
- **MySQL Connector**: Database connectivity
- **Google Cloud BigQuery**: Cloud data warehouse

### Architecture
```
Data Sources (MySQL/BigQuery) 
    ↓
Data Loading & Preprocessing
    ↓
Analytics Engine (Pandas + Scikit-learn)
    ↓
Visualization Layer (Plotly)
    ↓
Interactive Dashboard (Streamlit)
```

---

🎉 **Enjoy exploring your sales data with powerful analytics and insights!**

# Sales Dashboard Deployment Instructions for Streamlit Community Cloud

## 1. Prepare Your Repository
- Ensure the following files are present:
  - sales_dashboard.py (main app)
  - requirements.txt (Python dependencies)
  - .streamlit/secrets.toml (for MySQL credentials, not committed to GitHub)

## 2. Push to GitHub
- Create a new GitHub repository.
- Upload all your project files except .env and .streamlit/secrets.toml (these are for local use and secrets should be set in Streamlit Cloud UI).

## 3. Set Up Secrets in Streamlit Cloud
- Go to https://streamlit.io/cloud and sign in with GitHub.
- Click "New app", select your repo and branch, and set `sales_dashboard.py` as the main file.
- In the app settings, go to "Secrets" and add your MySQL credentials as:
  ```
  mysql_host = "your-mysql-host"
  mysql_user = "your-mysql-username"
  mysql_password = "your-mysql-password"
  mysql_database = "your-database-name"
  ```

## 4. Deploy
- Click "Deploy". Your app will be live at a public URL.

## 5. Usage
- Open the provided URL from any browser.
- The dashboard will connect to your MySQL database using the credentials from Streamlit secrets.

## Notes
- Do not commit sensitive credentials to GitHub.
- If you need to update dependencies, edit requirements.txt and redeploy.
- For troubleshooting, check the Streamlit Cloud logs.
