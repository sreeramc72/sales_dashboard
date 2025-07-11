import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector
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
import time


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

def add_essential_calculated_columns(df):
    # --- Discount Code Grouping ---
    if 'DiscountCode' in df.columns:
        # --- Discount Name to Group Mapping (from user list) ---
        DISCOUNT_NAME_TO_GROUP = {
            # Staff Meals & Others (updated list)
            'Total Discount (Staff Meals & Others)': [
                '10% - Corporate Discount','10% - Esaad Discount','10% Discount - AL BANDAR & ETIHAD STAFF','10% Loyal Customer','100 % Marketing','100 % staff Meal','100 AED - Voucher','100 AED GIFT VOUCHER','100 AED VOUCHER','100 Aed Voucher Discount','100% - Area Manager Discount','100% - Branch Complaint','100% - Branch Mistake','100% - Customer Care Discount','100% - Growth Discount','100% - Manager\'s Discount','100% - Marketing Discount','100% - Staff Meal Discount','100% - Training Discount','100% ACAI LUV','100% agent mistake','100% FOOD TASTING','100% Growth Discount','100% Growth\'s Discount','100% influencer discount','100% Manager\'s Discount','100% Managers Meal','100% Marketing','100% Marketing Aggregator Meal','100% marketing discount','100% Marketing Influencer','100% QSA','100% QSA DISCOUNT','100% Shi Meshwi Staff Meal','100% SOUP LUV','100% Staff Meal','100% Taste Panel Discount','100% Tawookji Managers Meal','100% Training Discount','100% Training Meal Discount','15% - Corporate Discount','15% Corporate Deal','150 AED - Gift Voucher','150 AED - Voucher','150 AED GIFT VOUCHER','20% - Corporate Discount','20% - Esaad Discount','20% - Mall Staff Discount','20% Corporate Discount','20% Corporate Order','20% Staff Meal','25% corporate','25% OFF - Sister Company/Hamper Discount','2nd Meal on Duty','30 % Essaad Corporate Discount','30% - Coalyard','30% - Coalyard Cafe','30% - discount','30% - Hokania Cafe','30% - Hookania Discount','30% Coalyard Cafe','30% OFF Hokania Cafe','30% OFF Padel Pro','30% Staff Meal','300 AED - VOUCHER','40% Social Brewer','5% Corporate Discount','50 % Discount Head office','50% Agent Mistake','50% OFF Sustainabilty Discount','50% Shi Meshwi Lava Brands','50% Shi Meshwi Staff Discount','70 % Top Management','70 HO Meal','70% - Staff Meal Discount','70% Mighty Slider Discount','70% Staff Meal','70% Tabkhet Salma Staff Meal','70% Tawookji Staff','70% Top Management','ACAI LUV QSA 100%','ADCB','ADNOC','AED','Agent Mistake','Agent Mistake Discount (100%)','Al Saada','Albandar & Etihad','Area Manager Discount','Area Manager Discount!','Bagel Luv 100% Training','Bagel Luv QSA 100%','BC60','Branch Complaint','Branch Complaint 100%','Cakery Staff Discount 40%','cancelled orders bad weather 100%','Chinese New Year 25%','Coal yard cafe Discount 30%','Coalyard 30%','Corporate - 20% discount','corporate 10%','Corporate 10% Discount','Corporate 15%','Corporate 15% Discount','Corporate 20%','Corporate 20% Discount','Corporate 20% Off','Corporate 20%Discount','Corporate 25% Discount','Corporate 30% Discount','corporate 40 %','Corporate Deals - 20% Discount','Corporate Discount - 20%','Corporate Discount - 25%','Corporate Orders','Corporate Orders 20%','Corporate Orders 25%','Corporate Orders 30%','Customer Care','Customer Care - 100%','Customer care 100% discount','customer care 50%','Emaar Group Employees Discount - 15%','Emirates Platinum - 25% Discount','Esaad','Esaad Discount 20%','FOMO QSA 100%','Growth - 100% Discount','HO Friday Meal','HO Friday Meal 100% discount','HO Meal','HO Meal 70% Discount','Hookania - 30% discount','Influencer 100%','lava brands','Lava Discount - 50%','Lava Discount 50%','Mall Staff','Mall Staff Disco','Manager Meal on Duty','Manager Meal On Duty 100%','Manager on duty meal 100% discount','Manager\'s Meal - 100 % Discount','Manager\'s Meal - 100% Discount','Manager\'s meal discount 100%','Marketing','Marketing - 100% discount','Marketing 100 %','Marketing 100%','Marketing 100% Discount','Off duty meal / 2nd meal on duty 30% discount','Padel Pro 30 %','promo Branch Complaint','QSA - PROMO DISCOUNT','R & I 100 % Discount','R & I Discount 100%','R&I Training 100%Discount','Social Brewer 40%','Social Brewer 45%','Social Brewer 45% discount','Staff Discount','Staff Meal 100 %','Staff meal discount','Staff Meal on Duty','Staff Meal On Duty - 100%','Staff Meal On Duty 100%','Staff on duty meal 100% discount','Staff on Duty Meal 70 %','step cafe 30% discount','Step Cafe 30% Discount 30%','step up 30%','Stuff Meal On Duty 100%','stuff meal on duty 100% Discount','Stuuf Meal on Duty 70%','Taste Panel','Taste Panel 100 %','Taste Panel 100%','Taste Panel 100% Discount','Taste Panel Discount','Test Order','TEST ORDER - 100% discount','Test order 100 %','Testing Orders','TGB Social Brewer 40%','Top Management','Top management / ho meal 70% discount','Top Management 70% Discount','Training 100% Discount','Training Department','training department 100%','Training Meal - 100% discount','Voucher - 50 AED'
            ],
            'Total Discount (Agg)': [
                '20158','26791','613627','625628','650060','654701','662535','10 % OFF','10% Talabat DineOut','10452sqkm','10AED off - talabat Rewards','15 % OFF','15 % OFF | PICKUP50','15 % OFF | SMILES30','15 % OFF | SMILES50','15% - Zomato Pro Discount','15% Talabat DineOut','15BCF','15BCG','15BFF','15CAKE','15CARI','15RAMADAN','20 % OFF','20 % OFF | COMBO100','20 % OFF | FLAT20','20 % OFF | HUNGRY30','20 % OFF | Item Level discount','20 % OFF | PICKUP50','20 % OFF | SMILES30','20 % OFF | SMILES50','20 % OFF | SMILES50 | Item Level discount','20 % OFF | WEEKEND50','20 % OFF | WEEKEND50 | Item Level discount','20EFT','20FEAST','20FEASTT','20FTU','20HLT','20KAY','20LUV','20MAN','20MSP','20YOU','20ZBT','25 % OFF','25 % OFF | COMBO100','25 % OFF | FLAT20','25 % OFF | HOMEFEAST','25 % OFF | HOTDEAL','25 % OFF | HUNGRY30','25 % OFF | Item Level discount','25 % OFF | PICKUP50','25 % OFF | PICKUP50 | Item Level discount','25 % OFF | SMILES30','25 % OFF | SMILES50','25 % OFF | SMILES50 | Item Level discount','25 % OFF | SMILES60','25 % OFF | SUPER30','25 % OFF | WEEKEND50','25% Discount','25% Discount - Entertainer','25BBK','25BGT','25EFT','25EPB','25FTU','25HSP','25KAY','25OBG','25PTC','25PTR','25TFT','25WPP','25WRP','25ZBT','30 % OFF','30 % OFF | BTS50','30 % OFF | COMBO100','30 % OFF | FLAT20','30 % OFF | HUNGRY30','30 % OFF | Item Level discount','30 % OFF | PICKUP50','30 % OFF | SMILES30','30 % OFF | SMILES50','30 AED - Careem','30% - Smiles Discount','30% - Zomato Discount','30% Careem Dine Out','30% Discount Deliveroo','30% off','30CR','30EHN','30EPB','30EVERY','30FAVES','30FTU','30HL','30SSR','30TGB','35 % OFF','35 % OFF | HUNGRY30','35 % OFF | PICKUP50','35 % OFF | SMILES30','40 % OFF','40 % OFF | SMILES30','50 % OFF','50 % OFF | PICKUP50','50 % OFF | SMILES50','50 % OFF | SUPER10','50% Lava Discount','50CWK','50DW','50PBT','50PNP','AC25','ACAI20','ADIB50','AL20','AL30','ALV20','ANX40','ASB50','ASIAN20','ASR30','AUS 15% discount','BA35','BAG20','BAK25','BAKE30','BB25','BBC20','BBK15','BBK25','BEM20','BFE20','BFE25','BFR25','BGL25','BIRD30','BL20','BL25','BM30','BMT20','BMT25','BNS30','BOGO','BOGO ENTERTAINER','BOGO20','BREAKFAST30','BURGER20','C50','CAKE15','CAKE20','CAKE50','CAKERY15','Cakery50','Careem - 10% Flat Discount','Careem - 10% MOV 20 Cap 30','Careem - 20% Discount','Careem - 20% Flat Discount','Careem - 20% MOV 20 Cap 30','Careem - 25% Discount','Careem - 25% Flat Discount','Careem - 25% MOV 20 Cap 30','Careem - 30% Discount','Careem - 30% Flat Discount','Careem - 40% Flat Discount','Careem - 50% MOV 40 Cap 30','Careem - 50% MOV40 Max AED30','Careem 20%','Careem 25%','Careem 30%','Careem 40%','Careem 50%','Careem Cap 30','CB100','CB50','Chain 630486','CHEEKY20','CHEEKY25','CKM20','CMG10','CO30','COFFEE30','COMBO100','COMBO100 | Item Level discount','CP50','CUP20','CUPP20','DAY10','DAY20','DAYY20','Deliveroo 20%','Deliveroo 20% Discount','Deliveroo 30 AED','Deliveroo Cap 40','dis_50','Disc_%20','Disc_10','Disc_15','Disc_20','Disc_20%','Disc_25','Disc_30','Disc_30 | Talabat_Gem','Disc_35','Disc_35%','Disc_40','Disc_50','Disc_50 | Talabat_Gem','Disc_52','Discount','Discount | Disc_52','Discount | Discount','Discount | Discount | Discount','Discount | Discount | Discount | Discount | Discount','DRM20','EAT20','EatEasy 20%','EatEasy 25%','EatEasy 30%','EatEasy 35%','EatEasy 40%','EATT20','EB25','EBK20','EFT20','EID50','EIDIYA','EL30','ELB30','Entertainer - 25%','Entertainer - BOGO','Entertainer 25%','Entertainer BOGO','Entertainer Offer','FAB15','FAR25','FAST50','FATA20','FATA25','FAYSAL20','FAZAA20','FB20','FB25','FCH25','FCH30','FEAST','FEAST10','FEAST20','FERN15','FERN20','FERN25','FERN35','FGR25','FIRST50','FIRSTBITE','FKT25','FLASH50','flat 30%','FLAT20','FLAT20 | Item Level discount','FMB20','FOMO20','FOMO25','FOMO30','FOMO50','FORYOU20','FORYOU50','FRIED30','FT35','FT50','FTU25','FTY20','GB30','GOCARI','Good Day','Good Day - BOGO','GOOD25','GOTYOU','GOTYOU.','HALFDEAL','HB25','HB30','HEA50','HERO50','HL30','HLT20','HOMEFEAST','HOMEFEAST | Item Level discount','HOTDEAL','HPK50ED','HUB15','HUNGRY30','IGET25','IKP25','Instashop - 20% Discount','Instashop - 20% Flat Discount','Instashop - 30% Flat Discount','Instashop - 35% Flat Discount','Instashop - 40% Flat Discount','Instashop 20%','Instashop 25%','Instashop 30%','Instashop 35%','Instashop 40%','Item Level discount','K20','KAAYK20','KAY10','KAY15','KAY20','KAY25','KAY30','KAY50','KAYK20','KAYK25','KAYK35','KAYK50','KAYKROO25','KAYY50','KB20','KB25','KBF25','KFP20','KK15','KK20','KK25','KK30','KK35','KK40','KK50','KKK50','KKRO20','KL20','KL30','KO20','KR25','KRO10','KRON20','LUV20','LUV25','MACHU20','MAN20','MANN20','MANO20','MANOUSHE50','MAQLUBA','MESH25','MESH30','MESHAWI50','MIGHT20','MNS10','MOUSSAKA','MS10','MS20','MS25','MSC10','MSC20','MSF25','MSP10','MSS20','MST10','MST20','MSTREET25','MSTT10','MSTT20','MTS10','NBD20','NEW30','NEW50','NEWCARI','NOON - 20% Flat Discount','NOON - 25% Flat Discount','NOON - 30% Flat Discount','NOON - 40% Flat Discount','Noon 20%','Noon 25%','Noon 30%','NOON 30% Discount - MOV 30 - Cap 20','Noon 40%','Noon 50%','NOON 50% Discount - MOV 30 - Cap 20','Noon Cap 20','NOT_MAPPED','NOT_MAPPED | 10AED off - talabat Rewards','NOT_MAPPED | Discount','NOT_MAPPED | Talabat_Gem','NS25','NU30','NU35','NU50','OBG30','OFF20','OFO6VYL2P','PASTA25','PASTA30','PASTA35','PB50','PC50','PICK40','PICK50','PICKN','PICKN50','PICKUP50','PICKUP50 | Item Level discount','PLUS20','PLUS25','PLUS30','PLUSMONDAY20','PLUSMONDAY20M','PLUSMONDAY30','PLUSMONDAY30DH','PNP50','POS_25','PPAIR50','PROMOTION','PROMOTION | 15CARI','PROMOTION | ADFOOD25','PROMOTION | CARIGO1','PROMOTION | CB50','PROMOTION | FAZAA20','PROMOTION | GET15AED','PROMOTION | YAHYA25','PROMOTION | ZNK10','PTR20','RAM20','RAMADAN20','Restaurant Discount','RNEW25','ROO25','RSB20','SA20','SAF20','SALUV25','SANDWICH20','SANDWICH25','SAUCE25','SAUCY50','SAVEU20','SAVUU20','SB15','SB25','SB50','SBT25','SCT15','SCUP20','SFIHA','SHI25','SHIM30','SHIME30','SLURP20','SLURP25','Smiles','SMILES30','SMILES30 | Item Level discount','SMILES50','SMILES50 | Item Level discount','SMILES60','SOC25','SOCIETY30','SOP20','SOP25','SOUL15','SOUP20','SOUP25','SOUP30','SP25','SP30','SPANAKOPITA','SPW30','SPZ15','string','string | string','string | string | string','string | string | string | string','STZ25','SUPER30','SURE50','SUSURU35','SWEET20','T30','TAKE20','TAKE25','Talabat - 20% Churned Discount - Cap 30','Talabat - 20% discount','Talabat - 20% discount, Cap 30aed','Talabat - 20% Flat Discount','Talabat - 20% New Customer Discount','Talabat - 20% New Customer Discount - Cap 30','Talabat - 25% discount','Talabat - 25% Flat Discount','Talabat - 30% discount','Talabat - 30% discount, Cap 30aed','Talabat - 30% Flat Discount','Talabat - 30% New Customer Discount','Talabat - 30% New Customer Discount, Cap 30','Talabat - 35% discount','Talabat - 35% Flat Discount','Talabat - 40% Flat Discount','Talabat - 50% discount, cap 30aed','Talabat - 50% Flash Sale, Cap 30aed','Talabat - Super Saver - 50% Discount, Cap 30','Talabat 15%','Talabat 20%','Talabat 20% Discount','Talabat 20% off','Talabat 25%','Talabat 30%','talabat 30%off','Talabat 35%','talabat 35% Discount','Talabat 40%','talabat 50%','Talabat Cap 30','Talabat Pro 15%','Talabat Regular 10%','Talabat_Gem','Talabat_Gem | Disc_50','Talabat_Gem | Disc_52','Talabna - 20% Discount','TGB25','TOULIN','TPB20','TPB25','TPC15','TPC20','TSBL50','TTF20','TTF25','TTZ20','UAE52','WEEKEND50','WEEKEND50 | Item Level discount','WFB25','WP20','WP25','WPD20','WPP20','WRA10','WRA20','WRAP20','WRAPP20','WRAPPED20','WRAPPED25','WRP10','WRP20','WRR50','WTC15','WTC25','WTC30','YAL10','YALLA10','YOURS20','YUM20','YUM25','YUM30','YUMMY30','ZATAAR25','Zomato Pro','2025-02-20'
                 ]
        }
        # Reverse mapping for fast lookup
        DISCOUNT_NAME_TO_GROUP_FLAT = {}
        for group, names in DISCOUNT_NAME_TO_GROUP.items():
            for name in names:
                DISCOUNT_NAME_TO_GROUP_FLAT[name.strip().upper()] = group

        def map_discount_group(code):
            if pd.isna(code):
                return 'None'
            code_str = str(code).strip().upper()
            return DISCOUNT_NAME_TO_GROUP_FLAT.get(code_str, 'Other')

        df['Discount_Group'] = df['DiscountCode'].apply(map_discount_group)
    """Add only essential calculated columns for faster performance"""
    if df.empty:
        return df
    
    # Check if net_sale has been calculated (it should be)
    if 'net_sale' not in df.columns:
        # New business logic: If Discount_Group is 'Total Discount (Staff Meals & Others)', use VAT-adjusted difference, else just GrossPrice/1.05
        if 'Discount_Group' in df.columns:
            df['net_sale'] = np.where(
                df['Discount_Group'] == 'Total Discount (Staff Meals & Others)',
                (df['GrossPrice'] / 1.05) - (df['Discount'] / 1.05),
                df['GrossPrice'] / 1.05
            )
        else:
            # Fallback to old logic if Discount_Group is missing
            df['net_sale'] = df['GrossPrice'] / 1.05
    
    # Financial metrics (essential only)
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        # Calculate profit margin
        df['Profit_Margin'] = ((df['net_sale'] - df['Discount']) / df['GrossPrice'] * 100).fillna(0)
        
        # Ensure discount percentage is calculated
        if 'Discount_Percentage' not in df.columns:
            df['Discount_Percentage'] = (df['Discount'] / df['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
    
    # Add minimum required time-based columns
    if 'ReceivedAt' in df.columns and 'DayOfWeek' not in df.columns:
        df['DayOfWeek'] = df['ReceivedAt'].dt.day_name()
    
    if 'Hour' not in df.columns and 'ReceivedAt' in df.columns:
        df['Hour'] = df['ReceivedAt'].dt.hour
    
    # Add Order_Size_Category for segmentation
    if 'Order_Size_Category' not in df.columns:
        df['Order_Size_Category'] = pd.cut(df['net_sale'], 
                                        bins=[0, 25, 75, 150, float('inf')],
                                        labels=['Small', 'Medium', 'Large', 'XLarge'])
    
    return df

def add_calculated_columns(df):
    # --- Discount Code Grouping ---
    if 'DiscountCode' in df.columns:
        DISCOUNT_CODE_GROUPS = {
            'EMP': 'Employee',
            'PROMO': 'Promotion',
            'LOYAL': 'Loyalty',
            'WELCOME': 'Welcome',
            'BIRTHDAY': 'Birthday',
            # Add more mappings as needed
        }
        def map_discount_group(code):
            if pd.isna(code):
                return 'None'
            for prefix, group in DISCOUNT_CODE_GROUPS.items():
                if str(code).upper().startswith(prefix):
                    return group
            return 'Other'
        df['Discount_Group'] = df['DiscountCode'].apply(map_discount_group)
    """Add calculated columns for enhanced analysis"""
    if df.empty:
        return df
    
    # Check if net_sale has been calculated (it should be)
    if 'net_sale' not in df.columns:
        # New business logic: If Discount_Group is 'Total Discount (Staff Meals & Others)', use VAT-adjusted difference, else just GrossPrice/1.05
        if 'Discount_Group' in df.columns:
            df['net_sale'] = np.where(
                df['Discount_Group'] == 'Total Discount (Staff Meals & Others)',
                (df['GrossPrice'] / 1.05) - (df['Discount'] / 1.05),
                df['GrossPrice'] / 1.05
            )
        else:
            # Fallback to old logic if Discount_Group is missing
            df['net_sale'] = df['GrossPrice'] / 1.05
    
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

@st.cache_data(ttl=3600)  # Cache for 1 hour (increased from 5 minutes)
def load_data_from_mysql(days_back=7):
    """Load data from MySQL database with optimized query for better performance"""
    try:
        # Show loading indicator
        with st.spinner(f"🔄 Loading data from MySQL for last {days_back} days..."):
            conn = mysql.connector.connect(
                host=st.secrets["MYSQL_HOST"],
                port=int(st.secrets.get("MYSQL_PORT", 3306)),
                user=st.secrets["MYSQL_USER"],
                password=st.secrets["MYSQL_PASSWORD"],
                database=st.secrets["MYSQL_DATABASE"]
            )
            
            # Optimized query: Select only required columns, use WHERE clause first
            query = f"""
            SELECT OrderID, CustomerName, ReceivedAt, GrossPrice, Discount, 
                   Delivery, Tips, VAT, Surcharge, Total, 
                   Channel, Brand, Location, PaymentMethod
            FROM sales_data 
            WHERE ReceivedAt >= DATE_SUB(NOW(), INTERVAL {days_back} DAY)
            ORDER BY ReceivedAt DESC
            """
            
            # Load data in chunks if it's large
            df = pd.read_sql(query, conn, parse_dates=['ReceivedAt'])
            conn.close()
        
        # Essential preprocessing only - defer other processing
        df['Date'] = df['ReceivedAt'].dt.date
        df['Hour'] = df['ReceivedAt'].dt.hour
        df['DayOfWeek'] = df['ReceivedAt'].dt.day_name()
        
        # Clean numeric columns (only essential ones)
        numeric_cols = ['GrossPrice', 'Discount', 'VAT', 'Delivery', 'Tips', 'Surcharge', 'Total']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate net_sale as a custom column in Python (not from database)
        # New business logic: If Discount_Group is 'Total Discount (Staff Meals & Others)', use VAT-adjusted difference, else just GrossPrice/1.05
        if 'Discount_Group' in df.columns:
            df['net_sale'] = np.where(
                df['Discount_Group'] == 'Total Discount (Staff Meals & Others)',
                (df['GrossPrice'] / 1.05) - (df['Discount'] / 1.05),
                df['GrossPrice'] / 1.05
            )
        else:
            df['net_sale'] = df['GrossPrice'] / 1.05
        
        # Add only the most essential calculated columns
        if 'GrossPrice' in df.columns and 'Discount' in df.columns:
            df['Discount_Percentage'] = (df['Discount'] / df['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def load_data_from_aws(days_back=7):
    """AWS data source placeholder - functionality coming soon"""
    st.warning("⚠️ AWS data source is not yet implemented. This feature is coming soon!")
    st.info("💡 For now, please use MySQL as your data source.")
    return pd.DataFrame()

def calculate_metrics(df):
    """Calculate key performance metrics"""
    if df.empty:
        return {}

    # Exclude 100% discounted orders (GrossPrice == Discount) from metrics
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

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

    # Exclude 100% discounted orders (GrossPrice == Discount) from trends
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

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

    # Exclude 100% discounted orders (GrossPrice == Discount) from customer analysis
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

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

    # Exclude 100% discounted orders (GrossPrice == Discount) from discount analysis
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

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
        st.warning("⚠️ No data available for pivot table analysis")
        return

    # Exclude 100% discounted orders (GrossPrice == Discount)
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

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
        else:
            st.warning("⚠️ Date column not found. Unable to generate predictions.")
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
            st.info(f"📊 **Smart Default**: Comparing exactly {second_latest_date} vs {latest_date} (2 most recent dates)")
        
        # IMPORTANT: Always filter to only show the selected dates, not the range between them
        selected_dates = [start_date, end_date]
        
        # Remove duplicates in case start_date == end_date
        selected_dates = list(set(selected_dates))
        
        # Filter data to only include the specific selected dates
        df_filtered = df[df['Date'].dt.date.isin(selected_dates)].copy()
        
        # Format the message based on whether one or two dates are selected
        if len(selected_dates) == 1:
            date_range_text = f"Showing data for {selected_dates[0]} only (single date)"
        else:
            # Sort dates for clearer display
            selected_dates_sorted = sorted(selected_dates)
            date_range_text = f"Showing data for exactly {selected_dates_sorted[0]} and {selected_dates_sorted[1]} only (not the range between)"
        
        # Additional filtering info
        excluded_info = ""
        if 'Discount_Percentage' in df.columns:
            total_orders = len(df)
            filtered_orders = len(df_filtered)
            hundred_percent_excluded = len(df[df['Discount_Percentage'] >= 100.0])
            if hundred_percent_excluded > 0:
                excluded_info = f" (excluding {hundred_percent_excluded:,} orders with 100% discount)"
        
        st.info(f"📊 Filtered data: {len(df_filtered):,} records {date_range_text}{excluded_info}")
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
        'Order_Size_Category', 'Discount_Category', 'Time_Period', 'Discount_Group'
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
            
            # Create individual pivot tables with explicit date sorting
            pivot_net_sale = pivot_data.pivot_table(
                values='net_sale',
                index='Channel',
                columns='Date',
                aggfunc='sum',
                fill_value=0
            )
            
            # Always sort date columns in ascending order (chronological)
            pivot_net_sale = pivot_net_sale.reindex(sorted(pivot_net_sale.columns), axis=1)
            
            pivot_discount = pivot_data.pivot_table(
                values='Discount',
                index='Channel',
                columns='Date',
                aggfunc='sum',
                fill_value=0
            )
            
            # Always sort date columns in ascending order (chronological)
            pivot_discount = pivot_discount.reindex(sorted(pivot_discount.columns), axis=1)
            
            pivot_discount_pct = pivot_data.pivot_table(
                values='Discount_Percentage',
                index='Channel',
                columns='Date',
                aggfunc='mean',
                fill_value=0
            )
            
            # Always sort date columns in ascending order (chronological)
            pivot_discount_pct = pivot_discount_pct.reindex(sorted(pivot_discount_pct.columns), axis=1)
            
            # Create the combined multi-level pivot table
            combined_pivot = pd.DataFrame()
            
            # IMPORTANT: Always sort dates in ascending order (chronological)
            dates = sorted(pivot_net_sale.columns)
            
            # Clear display about the date ordering
            if len(dates) > 1:
                st.info(f"📅 Date columns are ordered chronologically: {', '.join(str(d) for d in dates)}")
            
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
                # Ensure dates are sorted in ascending order (oldest to newest)
                sorted_dates = sorted(dates)
                first_date = str(sorted_dates[0])
                second_date = str(sorted_dates[1])
                
                # Calculate differences: Second - First (newer minus older)
                combined_pivot[(f'Δ {second_date} vs {first_date}', 'Net Sale Δ')] = (
                    combined_pivot[(second_date, 'Net Sale')] - combined_pivot[(first_date, 'Net Sale')]
                )
                
                combined_pivot[(f'Δ {second_date} vs {first_date}', 'Discount Δ')] = (
                    combined_pivot[(second_date, 'Discount')] - combined_pivot[(first_date, 'Discount')]
                )
                
                combined_pivot[(f'Δ {second_date} vs {first_date}', 'Discount % Δ')] = (
                    combined_pivot[(second_date, 'Discount %')] - combined_pivot[(first_date, 'Discount %')]
                )
                
                # Recreate columns to maintain proper order (chronological order)
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
                    (f'Δ {second_date} vs {first_date}', 'Net Sale Δ'),
                    (f'Δ {second_date} vs {first_date}', 'Discount Δ'),
                    (f'Δ {second_date} vs {first_date}', 'Discount % Δ')
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
                csv_discount = pivot_discount.to_csv()
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
                        analysis_dates = sorted(analysis_pivot_net.columns)  # Ensure chronological order
                        
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
                        combined_dates = sorted(combined_pivot_net.columns)  # Ensure chronological order
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
                    
                    with insight_col2:                    st.metric(
                        label="💸 Highest Discount Channel",
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
                # Create the pivot table
                pivot_table = df_filtered.pivot_table(
                    values=values,
                    index=rows,
                    columns=columns,
                    aggfunc=aggregation,
                    fill_value=0,
                    margins=True,
                    margins_name="Total"
                )
                # Sort columns if the column is Date to ensure chronological order
                if 'Date' in columns:
                    # Get non-Total columns and sort them
                    date_cols = [col for col in pivot_table.columns if col != "Total"]
                    other_cols = [col for col in pivot_table.columns if col == "Total"]
                    # Sort date columns in ascending order and append Total at the end
                    pivot_table = pivot_table[sorted(date_cols) + other_cols]
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
        except Exception as e:
            st.error(f"❌ Error creating pivot table: {str(e)}")
            st.write("Please try a different combination of dimensions and values.")
        # --- Debug Tab: Compare MySQL vs AWS Data ---
        # Ensure 'today' is defined for debug scope
        today_debug = pd.Timestamp.now().date()
        with st.expander('🛠️ Debug: Data Source Comparison (for selected date range)'):
            st.write('This section helps diagnose data issues and compare data sources.')
            # Calculate days_back for debug (same as sidebar logic)
            debug_days_back = (today_debug - start_date).days + 1
            df_mysql = load_data_from_mysql(debug_days_back)
            df_aws = load_data_from_aws(debug_days_back)
            st.write(f"MySQL: {len(df_mysql):,} rows, AWS: {len(df_aws):,} rows")
            if not df_mysql.empty:
                st.write('MySQL min/max ReceivedAt:', str(df_mysql['ReceivedAt'].min()), str(df_mysql['ReceivedAt'].max()))
            if not df_aws.empty:
                st.write('AWS min/max ReceivedAt:', str(df_aws['ReceivedAt'].min()), str(df_aws['ReceivedAt'].max()))
            # Show sample rows for the same day
            if not df_mysql.empty:
                sample_date = df_mysql['ReceivedAt'].dt.date.min()
                st.write(f"Sample date: {sample_date}")
                st.write('MySQL sample:')
                st.dataframe(df_mysql[df_mysql['ReceivedAt'].dt.date == sample_date][['OrderID','GrossPrice','Discount','ReceivedAt']].head(10))
            if not df_aws.empty:
                st.write('AWS sample:')
                st.dataframe(df_aws[df_aws['ReceivedAt'].dt.date == sample_date][['OrderID','GrossPrice','Discount','ReceivedAt']].head(10))
            else:
                st.write('AWS: No data available (feature coming soon)')
            # Check for nulls in key columns
            if not df_mysql.empty:
                st.write('MySQL nulls:', df_mysql[['GrossPrice','Discount','ReceivedAt']].isnull().sum())
            if not df_aws.empty:
                st.write('AWS nulls:', df_aws[['GrossPrice','Discount','ReceivedAt']].isnull().sum())
            else:
                st.write('AWS: No data to check (feature coming soon)')

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
    st.title("Growth Team Dashboard")
    st.markdown("""
        Welcome to the Sales Analytics Dashboard. Use the sidebar to filter or change the data source.
    """)
    

    # Sidebar controls
    with st.sidebar:
        st.subheader("Data Source Settings")
        
        # Data source selection
        data_source = st.selectbox(
            "Choose Data Source",
            options=["MySQL", "AWS (Coming Soon)"],
            index=0,
            help="Select your data source. AWS integration is coming soon!"
        )
        
        if data_source == "AWS (Coming Soon)":
            st.info("🚧 AWS integration is in development and will be available soon!")
        else:
            st.info("✅ Using MySQL as data source")

        # Date range filter
        st.subheader("Date Range Filter")
        today = datetime.now().date()
        default_start = today - timedelta(days=6)
        default_end = today
        date_range = st.date_input(
            "Select date range",
            value=(default_start, default_end),
            min_value=today - timedelta(days=60),
            max_value=today
        )
        # Ensure date_range is always a tuple (start, end)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = default_start
            end_date = default_end

        # Channel and Brand filters (always visible, robust to missing data)
        channel_options, brand_options = [], []
        df_for_filter = st.session_state.df if 'df' in st.session_state and st.session_state.df is not None else None
        if df_for_filter is not None and not df_for_filter.empty:
            if 'Channel' in df_for_filter.columns:
                channel_options = sorted([x for x in df_for_filter['Channel'].dropna().unique()])
            if 'Brand' in df_for_filter.columns:
                brand_options = sorted([x for x in df_for_filter['Brand'].dropna().unique()])
        selected_channels = st.multiselect("Filter by Channel", channel_options, default=channel_options, help="Filter data by sales channel.")
        if not channel_options:
            st.caption("No channel data available.")
        selected_brands = st.multiselect("Filter by Brand", brand_options, default=brand_options, help="Filter data by brand.")
        if not brand_options:
            st.caption("No brand data available.")

        # Calculate days_back for backward compatibility (for MySQL/AWS loaders)
        days_back = (today - start_date).days + 1
        refresh_button = st.button("Refresh Data")

        # Add a progress indicator in sidebar
        if 'loading_state' not in st.session_state:
            st.session_state.loading_state = "ready"

        # Show a loading indicator
        if st.session_state.loading_state == "loading":
            with st.spinner("Loading data..."):
                st.progress(0.75, "Processing data")

        # Show data source tips for better performance
        st.info("💡 **Tips**: Use fewer days for quicker loading. AWS integration coming soon!")
    
    # Session state to track if data has been loaded
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
    
    # Load data automatically on first run or when refresh is clicked
    if not st.session_state.data_loaded or refresh_button:
        try:
            # Set loading state
            st.session_state.loading_state = "loading"
            
            # Load data with optimized parameters
            loading_message = f"Loading optimized data for the last {days_back} days from {data_source}..."
            with st.spinner(loading_message):
                if data_source == "MySQL":
                    df = load_data_from_mysql(days_back)
                elif data_source == "AWS (Coming Soon)":
                    df = load_data_from_aws(days_back)
                else:
                    # Default to MySQL for any other case
                    df = load_data_from_mysql(days_back)
                
                if not df.empty:
                    # Apply additional calculated columns only after successful load
                    # but only for columns needed in the active view
                    df = add_essential_calculated_columns(df)
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df):,} records from {data_source}.")
                else:
                    if data_source == "AWS (Coming Soon)":
                        st.info("ℹ️ AWS data source is not yet available. Please select MySQL to view data.")
                    else:
                        st.warning(f"⚠️ No data loaded from {data_source}. Please check your connection settings.")
            
            # Reset loading state
            st.session_state.loading_state = "ready"
            
        except Exception as e:
            # Reset loading state on error
            st.session_state.loading_state = "ready"
            st.error(f"Error loading data: {str(e)}")
    
    # Get the dataframe from session state
    df = st.session_state.df

    # --- Date filter for in-memory data (UI only, does not reload from DB) ---
    if df is not None and not df.empty:
        # Only filter if Date column exists
        if 'Date' in df.columns:
            # Ensure Date is datetime.date
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            # Apply Channel filter
            if 'Channel' in df.columns and 'selected_channels' in locals() and selected_channels:
                filtered_df = filtered_df[filtered_df['Channel'].isin(selected_channels)]
            # Apply Brand filter
            if 'Brand' in df.columns and 'selected_brands' in locals() and selected_brands:
                filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
        else:
            filtered_df = df
    else:
        filtered_df = df

    if filtered_df is not None and not filtered_df.empty:
        tabs = st.tabs([
            "📊 Sales Trends",
            "👥 Customer Behavior",
            "🎫 Discount Performance",
            "🔮 Sales Predictions",
            "⚖️ Comparative Analysis",
            "🎮 Shady's Command Center",
            "📋 Data Overview"
        ])

        # 1. Sales Trends Tab
        with tabs[0]:
            st.header("📊 Sales Trends")
            st.write("Daily revenue and order trends, hourly and day-of-week patterns, peak sales periods.")

            # Display key metrics at the top
            metrics = calculate_metrics(filtered_df)

            # Create columns for metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Revenue", f"${metrics['total_revenue']:,.2f}")
            with metric_cols[1]:
                st.metric("Total Orders", f"{metrics['total_orders']:,}")
            with metric_cols[2]:
                st.metric("Avg Order Value", f"${metrics['avg_order_value']:.2f}")
            with metric_cols[3]:
                st.metric("Discount Rate", f"{metrics['discount_rate']:.1f}%")

            # Display the sales trends chart
            sales_chart = create_sales_trends_chart(filtered_df)
            st.plotly_chart(sales_chart, use_container_width=True)
        
        # 2. Customer Behavior Tab
        with tabs[1]:
            st.header("👥 Customer Behavior")
            st.write("Customer segmentation, top customers, customer lifetime value.")

            # Display customer behavior metrics
            customer_cols = st.columns(3)
            with customer_cols[0]:
                st.metric("Total Customers", f"{metrics['total_customers']:,}")
            with customer_cols[1]:
                customer_metrics = filtered_df.groupby('CustomerName').size()
                repeat_customers = customer_metrics[customer_metrics > 1].count()
                st.metric("Repeat Customers", f"{repeat_customers:,}")
            with customer_cols[2]:
                repeat_rate = (repeat_customers / metrics['total_customers'] * 100) if metrics['total_customers'] > 0 else 0
                st.metric("Repeat Rate", f"{repeat_rate:.1f}%")

            # Display customer behavior chart and data
            behavior_chart, customer_data = create_customer_behavior_analysis(filtered_df)
            st.plotly_chart(behavior_chart, use_container_width=True)

            # Show top customers
            st.subheader("🏆 Top Customers")
            top_customers = customer_data.sort_values('Total_Spent', ascending=False).head(10)
            st.dataframe(top_customers[['CustomerName', 'Total_Spent', 'Order_Count', 'Avg_Order_Value', 'Segment']], use_container_width=True)
        
        # 3. Discount Performance Tab
        with tabs[2]:
            st.header("🎫 Discount Performance")
            st.write("Discount effectiveness, code performance, impact on sales.")

            # Display discount metrics
            discount_cols = st.columns(3)
            with discount_cols[0]:
                st.metric("Total Discount", f"${metrics['total_discount']:,.2f}")
            with discount_cols[1]:
                discount_rate = metrics['discount_rate']
                st.metric("Discount Rate", f"{discount_rate:.1f}%")
            with discount_cols[2]:
                discounted_orders = len(filtered_df[filtered_df['Discount'] > 0])
                discount_order_pct = (discounted_orders / metrics['total_orders'] * 100) if metrics['total_orders'] > 0 else 0
                st.metric("Orders with Discount", f"{discount_order_pct:.1f}%")

            # Display discount performance chart
            discount_chart = create_discount_performance_analysis(filtered_df)
            st.plotly_chart(discount_chart, use_container_width=True)
        
        # 4. Sales Predictions Tab
        with tabs[3]:
            st.header("🔮 Sales Predictions")
            st.write("30-day forecasts, model comparison, business planning insights.")
            
            # Show predictions only if enough data is available
            if len(df) >= 30:
                prediction_chart, model_metrics = create_sales_predictions(df)
                
                # Display model metrics
                st.subheader("Model Performance")
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    st.write("**Linear Regression**")
                    st.metric("MAE", f"{model_metrics['Linear Regression']['MAE']:.2f}")
                    st.metric("R² Score", f"{model_metrics['Linear Regression']['R²']:.3f}")
                
                with metric_cols[1]:
                    st.write("**Random Forest**")
                    st.metric("MAE", f"{model_metrics['Random Forest']['MAE']:.2f}")
                    st.metric("R² Score", f"{model_metrics['Random Forest']['R²']:.3f}")
                
                # Display prediction chart
                st.plotly_chart(prediction_chart, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient data for predictions. Need at least 30 days of data.")
        
        # 5. Comparative Analysis Tab with Pivot Tables
        with tabs[4]:
            st.header("⚖️ Comparative Analysis")
            st.write("Performance across channels, location-based sales, payment method preferences.")
            
            # Display comparison chart first
            comparison_chart = create_comparison_analysis(df)
            st.plotly_chart(comparison_chart, use_container_width=True)
        
        # 6. Command Centre Tab (Pivot Table)
        with tabs[5]:
            st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🎮 Shady's Command Center - Daily Analytical Hub</h1>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin-bottom: 20px;'>
                Your central hub for interactive analysis, decision making, and performance monitoring.
                Use the pivot tables below to dive deep into your data.
            </div>
            """, unsafe_allow_html=True)
            
            # Add a horizontal divider
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Call the pivot table analysis function
            create_pivot_table_analysis(df)
        
        # 7. Data Overview Tab
        with tabs[6]:
            st.header("📋 Data Overview")
            st.write("Data quality metrics, export as CSV, inspect raw data.")
            
            # Data quality metrics
            quality_cols = st.columns(3)
            with quality_cols[0]:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", f"{missing_values:,}")
            with quality_cols[1]:
                duplicate_rows = df.duplicated().sum()
                st.metric("Duplicate Rows", f"{duplicate_rows:,}")
            with quality_cols[2]:
                date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
                st.metric("Date Range", date_range)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            # Export data button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data as CSV",
                data=csv,
                file_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("Please select options and click 'Refresh Data' to begin or wait for automatic data loading.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log the error but don't show WebSocket errors to users
        if 'WebSocketClosedError' not in str(e) and 'Stream is closed' not in str(e):
            st.error(f"An error occurred: {str(e)}")
        # For WebSocket errors, just log and continue
        pass
