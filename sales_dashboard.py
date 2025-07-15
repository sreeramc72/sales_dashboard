# --- UI Changes Block ---
# Custom Streamlit UI tweaks for dashboard appearance
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Section header styling */
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    /* Metric card styling */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    /* Make all tables horizontally scrollable and readable */
    div[data-testid="stHorizontalBlock"] table {
        word-break: break-word;
        min-width: 1200px;
        font-size: 1.05em;
    }
    div[data-testid="stHorizontalBlock"] th, div[data-testid="stHorizontalBlock"] td {
        padding: 6px 10px;
    }
    /* Make all markdown tables scrollable */
    .scrollable-table {
        overflow-x: auto;
        margin-bottom: 1em;
    }
    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
# --- AWS MySQL Connection Test Utility ---
def test_aws_mysql_connection():
    import os
    from sqlalchemy.exc import SQLAlchemyError
    # Load from environment variables (from .env via dotenv)
    endpoint = os.getenv("MySQL_HOST") or os.getenv("MYSQL_HOST")
    db_name = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQL_DB")
    username = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    port = int(os.getenv("MYSQL_PORT") or 3306)
    try:
        engine = get_aws_mysql_engine(endpoint, db_name, username, password, port)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.sidebar.success(f"AWS MySQL connection successful! Host: {endpoint}")
    except SQLAlchemyError as e:
        st.sidebar.error(f"AWS MySQL connection failed: {e}")
    except Exception as e:
        st.sidebar.error(f"Unexpected error: {e}")


import os
import streamlit as st

# --- Third-party imports (for AWS MySQL helper) ---
from sqlalchemy import create_engine, text
import sqlalchemy
from sqlalchemy.engine import URL

# Helper function to connect to AWS MySQL
def get_aws_mysql_engine(endpoint, db_name, username, password, port=3306):
    url = URL.create(
        drivername="mysql+pymysql",
        username=username,
        password=password,
        host=endpoint,
        port=port,
        database=db_name
    )
    engine = sqlalchemy.create_engine(url)
    return engine

# ...existing code...

# --- Data Source Selection ---

# Add a button to test AWS MySQL connection using .env credentials
if st.sidebar.button("Test AWS MySQL Connection (.env)"):
    test_aws_mysql_connection()

st.sidebar.header("Data Source Selection")
data_source = st.sidebar.selectbox(
    "Choose Data Source:",
    ["Local MySQL", "AWS MySQL"],
    index=1  # AWS MySQL is default
)

if data_source == "Local MySQL":
    st.sidebar.success("Using Local MySQL (from .env)")
    db_host = os.getenv("MYSQL_HOST")
    db_name = os.getenv("MYSQL_DATABASE")
    db_user = os.getenv("MYSQL_USER")
    db_pass = os.getenv("MYSQL_PASSWORD")
    db_port = int(os.getenv("MYSQL_PORT") or 3306)
else:
    st.sidebar.success("Using AWS MySQL (from st.secrets)")
    db_host = st.secrets["MYSQL_HOST"]
    db_name = st.secrets["MYSQL_DB"]
    db_user = st.secrets["MYSQL_USER"]
    db_pass = st.secrets["MYSQL_PASSWORD"]
    db_port = int(st.secrets["MYSQL_PORT"])

# Create a single engine for all database operations
engine = get_aws_mysql_engine(
    endpoint=db_host,
    db_name=db_name,
    username=db_user,
    password=db_pass,
    port=db_port
)
# Use 'engine' for all database operations below
import sqlalchemy
import pandas as pd
from sqlalchemy.engine import URL

# Helper function to connect to AWS MySQL
def get_aws_mysql_engine(endpoint, db_name, username, password, port=3306):
    url = URL.create(
        drivername="mysql+pymysql",
        username=username,
        password=password,
        host=endpoint,
        port=port,
        database=db_name
    )
    engine = sqlalchemy.create_engine(url)
    return engine

# Example usage:
# aws_engine = get_aws_mysql_engine(
#     endpoint="your-aws-endpoint",
#     db_name="your-db-name",
#     username="your-username",
#     password="your-password"
# )
# df_aws = pd.read_sql("SELECT * FROM your_table", aws_engine)
    # BigQuery loader and debug UI removed as per user request
from deep_dive_discount_performance_insights import deep_dive_discount_performance_insights

# --- Standard library imports ---

# --- Utility: Sanitize DataFrame for Streamlit Arrow compatibility ---
def sanitize_for_streamlit(df):
    """Convert all object columns to string for Streamlit Arrow compatibility."""
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].apply(lambda x: str(x) if not (isinstance(x, (int, float, bool, type(None), pd.Timestamp))) else x)
            if df[col].dtype == 'O':
                df[col] = df[col].astype(str)
    return df
import os
import time
import warnings
import logging
import pathlib

# --- Third-party imports ---
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector
from sqlalchemy import create_engine, text
# ...existing code...

import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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
import pathlib
# Load environment variables and debug display

dotenv_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path, override=True)

# --- Debug: Show loaded MySQL environment variables in sidebar ---
# Debug section removed as SQL is now loading correctly

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
                '10% - Corporate Discount','10% - Esaad Discount','10% Discount - AL BANDAR & ETIHAD STAFF','10% Loyal Customer','100 % Marketing','100 % staff Meal','100 AED - Voucher','100 AED GIFT VOUCHER','100 AED VOUCHER','100 Aed Voucher Discount','100% - Area Manager Discount','100% - Branch Complaint','100% - Branch Mistake','100% - Customer Care Discount','100% - Growth Discount','100% - Manager\'s Discount','100% - Marketing Discount','100% - Staff Meal Discount','100% - Training Discount','100% ACAI LUV','100% agent mistake','100% FOOD TASTING','100% Growth Discount','100% Growth\'s Discount','100% influencer discount','100% Manager\'s Discount','100% Managers Meal','100% Marketing','100% Marketing Aggregator Meal','100% marketing discount','100% Marketing Influencer','100% QSA','100% QSA DISCOUNT','100% Shi Meshwi Staff Meal','100% SOUP LUV','100% Staff Meal','100% Taste Panel Discount','100% Tawookji Managers Meal','100% Training Discount','100% Training Meal Discount','15% - Corporate Discount','15% Corporate Deal','150 AED - Gift Voucher','150 AED - Voucher','150 AED GIFT VOUCHER','20% - Corporate Discount','20% - Esaad Discount','20% - Mall Staff Discount','20% Corporate Discount','20% Corporate Order','20% Staff Meal','25% corporate','25% OFF - Sister Company/Hamper Discount','2nd Meal on Duty','30 % Essaad Corporate Discount','30% - Coalyard','30% - Coalyard Cafe','30% - discount','30% - Hokania Cafe','30% - Hookania Discount','30% OFF Hokania Cafe','30% OFF Hookania Cafe','30% Coalyard Cafe','30% OFF Padel Pro','30% Staff Meal','300 AED - VOUCHER','40% Social Brewer','5% Corporate Discount','50 % Discount Head office','50% Agent Mistake','50% OFF Sustainabilty Discount','50% Shi Meshwi Lava Brands','50% Shi Meshwi Staff Discount','70 % Top Management','70 HO Meal','70% - Staff Meal Discount','70% Mighty Slider Discount','70% Staff Meal','70% Tabkhet Salma Staff Meal','70% Tawookji Staff','70% Top Management','ACAI LUV QSA 100%','ADCB','ADNOC','AED','Agent Mistake','Agent Mistake Discount (100%)','Al Saada','Albandar & Etihad','Area Manager Discount','Area Manager Discount!','Bagel Luv 100% Training','Bagel Luv QSA 100%','BC60','Branch Complaint','Branch Complaint 100%','Cakery Staff Discount 40%','cancelled orders bad weather 100%','Chinese New Year 25%','Coal yard cafe Discount 30%','Coalyard 30%','Corporate - 20% discount','corporate 10%','Corporate 10% Discount','Corporate 15%','Corporate 15% Discount','Corporate 20%','Corporate 20% Discount','Corporate 20% Off','Corporate 20%Discount','Corporate 25% Discount','Corporate 30% Discount','corporate 40 %','Corporate Deals - 20% Discount','Corporate Discount - 20%','Corporate Discount - 25%','Corporate Orders','Corporate Orders 20%','Corporate Orders 25%','Corporate Orders 30%','Customer Care','Customer Care - 100%','Customer care 100% discount','customer care 50%','Emaar Group Employees Discount - 15%','Emirates Platinum - 25% Discount','Esaad','Esaad Discount 20%','FOMO QSA 100%','Growth - 100% Discount','HO Friday Meal','HO Friday Meal 100% discount','HO Meal','HO Meal 70% Discount','Hookania - 30% discount','Influencer 100%','lava brands','Lava Discount - 50%','Lava Discount 50%','Mall Staff','Mall Staff Disco','Manager Meal on Duty','Manager Meal On Duty 100%','Manager on duty meal 100% discount','Manager\'s Meal - 100 % Discount','Manager\'s Meal - 100% Discount','Manager\'s meal discount 100%','Marketing','Marketing - 100% discount','Marketing 100 %','Marketing 100%','Marketing 100% Discount','Off duty meal / 2nd meal on duty 30% discount','Padel Pro 30 %','promo Branch Complaint','QSA - PROMO DISCOUNT','R & I 100 % Discount','R & I Discount 100%','R&I Training 100%Discount','Social Brewer 40%','Social Brewer 45%','Social Brewer 45% discount','Staff Discount','Staff Meal 100 %','Staff meal discount','Staff Meal on Duty','Staff Meal On Duty - 100%','Staff Meal On Duty 100%','Staff on duty meal 100% discount','Staff on Duty Meal 70 %','step cafe 30% discount','Step Cafe 30% Discount 30%','step up 30%','Stuff Meal On Duty 100%','stuff meal on duty 100% Discount','Stuuf Meal on Duty 70%','Taste Panel','Taste Panel 100 %','Taste Panel 100%','Taste Panel 100% Discount','Taste Panel Discount','Test Order','TEST ORDER - 100% discount','Test order 100 %','Testing Orders','TGB Social Brewer 40%','Top Management','Top management / ho meal 70% discount','Top Management 70% Discount','Training 100% Discount','Training Department','training department 100%','Training Meal - 100% discount','Voucher - 50 AED','30% OFF Hokania Cafe','30% - Hookania Discount','Discount','Discount | Discount','Discount | Discount | Discount','Discount | Discount | Discount | Discount','Discount | Talabat_Gem','Discount - 100%','Discount - 15%','Discount - 20%','Discount - 25%','Discount - 30%','Discount - 50%','Discount - 70%','Discount Code Grouping',
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
        
        # Add new business-specific discount calculation
        # Formula: if(DiscountCode='Total Discount (StaffMeal&Others)', 0, 
        #              if(AND(Channel='Talabat', Brand='Manoushe Street'), (Discount/1.05)/2, Discount/1.05))
        if 'Discount_Group' in df.columns:
            df['Calculated_Discount'] = np.where(
                df['Discount_Group'] == 'Total Discount (Staff Meals & Others)',
                0,  # Staff meals & others = 0 discount impact
                np.where(
                    (df.get('Channel', '') == 'Talabat') & (df.get('Brand', '') == 'Manoushe Street'),
                    (df['Discount'] / 1.05) / 2,  # Talabat + Manoushe Street = VAT-adjusted discount divided by 2
                    df['Discount'] / 1.05  # All other cases = VAT-adjusted discount
                )
            )
        else:
            # Fallback if Discount_Group is not available
            df['Calculated_Discount'] = np.where(
                (df.get('Channel', '') == 'Talabat') & (df.get('Brand', '') == 'Manoushe Street'),
                (df['Discount'] / 1.05) / 2,  # Talabat + Manoushe Street = VAT-adjusted discount divided by 2
                df['Discount'] / 1.05  # All other cases = VAT-adjusted discount
            )
    
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
    df['Profit_Margin'] = ((df['net_sale'] - df['Calculated_Discount']) / df['GrossPrice'] * 100).fillna(0)
    df['Discount_Percentage'] = (df['Calculated_Discount'] / df['GrossPrice'] * 100).fillna(0)
    df['Revenue_After_Delivery'] = df['net_sale'] - df['Delivery']
    df['Total_Fees'] = df['Delivery'] + df['Surcharge'] + df['VAT']
    df['Order_Profitability'] = df['net_sale'] - df['Calculated_Discount'] - df['Total_Fees']
    
    # Add new business-specific discount calculation
    # Formula: if(DiscountCode='Total Discount (StaffMeal&Others)', 0, 
    #              if(AND(Channel='Talabat', Brand='Manoushe Street'), (Discount/1.05)/2, Discount/1.05))
    if 'Discount_Group' in df.columns:
        df['Calculated_Discount'] = np.where(
            df['Discount_Group'] == 'Total Discount (Staff Meals & Others)',
            0,  # Staff meals & others = 0 discount impact
            np.where(
                (df.get('Channel', '') == 'Talabat') & (df.get('Brand', '') == 'Manoushe Street'),
                (df['Discount'] / 1.05) / 2,  # Talabat + Manoushe Street = VAT-adjusted discount divided by 2
                df['Discount'] / 1.05  # All other cases = VAT-adjusted discount
            )
        )
    else:
        # Fallback if Discount_Group is not available
        df['Calculated_Discount'] = np.where(
            (df.get('Channel', '') == 'Talabat') & (df.get('Brand', '') == 'Manoushe Street'),
            (df['Discount'] / 1.05) / 2,  # Talabat + Manoushe Street = VAT-adjusted discount divided by 2
            df['Discount'] / 1.05  # All other cases = VAT-adjusted discount
        )
    
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
def load_data_from_mysql(days_back=7, max_retries=3):
    """Load data from MySQL database with optimized query for better performance
    
    Args:
        days_back: Number of days of data to fetch
        max_retries: Maximum number of connection retry attempts
        
    Returns:
        DataFrame with sales data or empty DataFrame on error
    """
    
    # Initialize retry counter
    retry_count = 0
    connection_timeout = 10  # seconds
    
    df = pd.DataFrame()
    while retry_count < max_retries:
        try:
            # Show loading indicator with retry information if applicable
            retry_msg = f" (Attempt {retry_count + 1}/{max_retries})" if retry_count > 0 else ""
            with st.spinner(f"🔄 Loading data from MySQL for last {days_back} days{retry_msg}..."):
                # Log connection attempt (not exposing credentials)
                st.info(f"Connecting to MySQL database at {os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT', '3306')}")
                # Create connection with timeout
                conn = mysql.connector.connect(
                    host=os.getenv("MYSQL_HOST"),
                    port=int(os.getenv("MYSQL_PORT", 3306)),
                    user=os.getenv("MYSQL_USER"),
                    password=os.getenv("MYSQL_PASSWORD"),
                    database=os.getenv("MYSQL_DATABASE"),
                    connection_timeout=connection_timeout
                )
                # Optimized query: Select only required columns, use WHERE clause first
                query = f"""
                SELECT OrderID, CustomerName, Telephone, ReceivedAt, GrossPrice, Discount, 
                       Delivery, Tips, VAT, Surcharge, Total, 
                       Channel, Brand, Location, PaymentMethod
                FROM sales_data 
                WHERE ReceivedAt >= DATE_SUB(NOW(), INTERVAL {days_back} DAY)
                ORDER BY ReceivedAt DESC
                """
                # Load data in chunks if it's large
                st.info(f"Running query to fetch {days_back} days of sales data...")
                df = pd.read_sql(query, conn, parse_dates=['ReceivedAt'])
                conn.close()
                # Log successful connection
                st.success(f"Successfully connected to MySQL and retrieved {len(df)} records!")
                # Essential preprocessing only - defer other processing
                df['Date'] = df['ReceivedAt'].dt.date
                df['Hour'] = df['ReceivedAt'].dt.hour
                df['DayOfWeek'] = df['ReceivedAt'].dt.day_name()
                # Ensure Telephone column is string (for display and grouping)
                if 'Telephone' in df.columns:
                    df['Telephone'] = df['Telephone'].astype(str)
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
                # Break out of retry loop on success
                break
        except (mysql.connector.Error, pd.io.sql.DatabaseError) as db_err:
            # Database-specific errors (connection, query issues)
            error_message = str(db_err)
            st.error(f"Database error: {error_message}")
            # Check for specific connection errors
            if "Access denied" in error_message:
                st.error("⚠️ Authentication failed. Please check your database credentials.")
            elif "Can't connect" in error_message or "Connection refused" in error_message:
                st.error("⚠️ Cannot connect to database server. Please check if the server is running and accessible.")
            elif "Unknown database" in error_message:
                st.error("⚠️ Database does not exist. Please check your database name.")
            # Increment retry counter and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                st.warning(f"Retrying connection in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)  # Wait 5 seconds before retry
            else:
                st.error(f"Failed to connect after {max_retries} attempts. Please check your database configuration.")
                # Return empty DataFrame if all retries failed
                return pd.DataFrame()
        except Exception as e:
            # Handle other general exceptions
            st.error(f"Error loading data from MySQL: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                st.warning(f"Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)
            else:
                st.error("All retry attempts failed. Please check logs for details.")
                return pd.DataFrame()
    # This will only execute if all retries failed without raising an exception
    return df



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
        'total_discount': df['Calculated_Discount'].sum(),
        'discount_rate': (df['Calculated_Discount'].sum() / df['GrossPrice'].sum()) * 100 if df['GrossPrice'].sum() > 0 else 0,
        'avg_delivery_fee': df['Delivery'].mean(),
        'total_tips': df['Tips'].sum()
    }

    return metrics

# --- Custom Sales Prediction for Shady's Command Center ---
def calculate_custom_sales_prediction(df):
    """
    Custom prediction: total net_sale (whole day sale) of the second latest day
    + difference of net_sale till the latest update of the latest day.
    """
    if df.empty or 'Date' not in df.columns or 'net_sale' not in df.columns:
        return None, None, None

    # Get unique sorted dates (ascending)
    unique_dates = sorted(df['Date'].unique())
    if len(unique_dates) < 2:
        return None, None, None

    # Latest and second latest day
    latest_day = unique_dates[-1]
    second_latest_day = unique_dates[-2]


    # Net sale for second latest day (whole day)
    net_sale_second_latest = df[df['Date'] == second_latest_day]['net_sale'].sum()


    # For the latest day, get the max ReceivedAt (latest update)
    if 'ReceivedAt' in df.columns:
        latest_day_df = df[df['Date'] == latest_day]
        max_time = latest_day_df['ReceivedAt'].max()
        # Net sale till the latest update (so far today)
        net_sale_latest_so_far = latest_day_df[latest_day_df['ReceivedAt'] <= max_time]['net_sale'].sum()
        # For difference, get net_sale up to the same time on the previous day (if exists)
        prev_day_df = df[df['Date'] == second_latest_day]
        prev_day_same_time = prev_day_df[prev_day_df['ReceivedAt'] <= max_time]
        net_sale_prev_day_same_time = prev_day_same_time['net_sale'].sum()
        # Difference so far today vs same time yesterday
        net_sale_diff = net_sale_latest_so_far - net_sale_prev_day_same_time
    else:
        net_sale_diff = 0

    # Prediction: yesterday's (second latest) full day + (today till now - yesterday till same time)
    prediction = net_sale_second_latest + net_sale_diff
    return prediction, net_sale_second_latest, net_sale_diff

    # For the latest day, get the max ReceivedAt (latest update)
    if 'ReceivedAt' in df.columns:
        latest_day_df = df[df['Date'] == latest_day]
        max_time = latest_day_df['ReceivedAt'].max()
        # Net sale till the latest update (so far today)
        net_sale_latest_so_far = latest_day_df['net_sale'].sum()
        # For difference, get net_sale up to the same time on the previous day (if exists)
        prev_day_df = df[df['Date'] == second_latest_day]
        prev_day_same_time = prev_day_df[prev_day_df['ReceivedAt'] <= max_time]
        net_sale_prev_day_same_time = prev_day_same_time['net_sale'].sum()
        # Difference so far today vs same time yesterday
        net_sale_diff = net_sale_latest_so_far - net_sale_prev_day_same_time
    else:
        net_sale_diff = 0

    # Prediction: yesterday's (second latest) full day + difference so far today vs same time yesterday
    prediction = net_sale_second_latest + net_sale_diff
    return prediction, net_sale_second_latest, net_sale_diff

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
        'Calculated_Discount': 'sum'
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
        go.Histogram(x=df['Calculated_Discount'], name='Discount Distribution',
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

    # Exclude 100% discounted orders (GrossPrice == Discount) from customer analysis (staff orders)
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

    # Customer analysis (with Telephone and Address if available)
    group_cols = ['CustomerName']
    if 'Telephone' in df.columns:
        group_cols.append('Telephone')
    if 'Address' in df.columns:
        group_cols.append('Address')
    customer_stats = df.groupby(group_cols).agg({
        'net_sale': ['sum', 'mean', 'count'],
        'Calculated_Discount': 'sum',
        'ReceivedAt': ['min', 'max']
    }).round(2)
    # Flatten columns
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
    
    # Show customer table with all available info (CustomerName, Telephone, Address)
    display_cols = ['CustomerName']
    if 'Telephone' in customer_stats.columns:
        display_cols.append('Telephone')
    if 'Address' in customer_stats.columns:
        display_cols.append('Address')
    display_cols += ['Total_Spent', 'Order_Count', 'Total_Discount', 'Customer_Lifetime_Days', 'Segment']
    st.subheader('👤 Customer Table (Name, Telephone, Address)')
    # Fix Arrow serialization error: ensure all columns are string or numeric (esp. Address, Telephone)
    # Fix Arrow serialization error: ensure all columns are string, numeric, or datetime (esp. Address, Telephone)
    for col in display_cols:
        if customer_stats[col].dtype == 'O':
            # Convert lists, dicts, sets, or other objects to string
            customer_stats[col] = customer_stats[col].apply(lambda x: str(x) if not (isinstance(x, (int, float, bool, type(None), pd.Timestamp))) else x)
        # If still object dtype, force to string
        if customer_stats[col].dtype == 'O':
            customer_stats[col] = customer_stats[col].astype(str)
    st.dataframe(sanitize_for_streamlit(customer_stats[display_cols]), use_container_width=True)
    return fig, customer_stats

def create_discount_performance_analysis(df):
    """Analyze discount performance and effectiveness"""
    if df.empty:
        return None, None, None, None, None

    # Exclude 100% discounted orders (GrossPrice == Discount) from discount analysis (staff orders)
    if 'GrossPrice' in df.columns and 'Discount' in df.columns:
        df = df[df['GrossPrice'] != df['Discount']].copy()

    # Discount analysis
    discount_df = df[df['Calculated_Discount'] > 0].copy()
    if discount_df.empty:
        return None, None, None, None, None

    # Group by discount ranges
    discount_df['Discount_Range'] = pd.cut(discount_df['Calculated_Discount'], 
                                         bins=[0, 10, 25, 50, 100, float('inf')],
                                         labels=['1-10', '11-25', '26-50', '51-100', '100+'])

    # --- Contribution Margin Calculation ---
    # Assume Profit_Margin or Order_Profitability exists, else estimate CM as net_sale - Calculated_Discount - Delivery - VAT - Surcharge
    if 'Profit_Margin' in discount_df.columns:
        discount_df['CM'] = discount_df['Profit_Margin']
    elif 'Order_Profitability' in discount_df.columns:
        discount_df['CM'] = discount_df['Order_Profitability']
    else:
        for col in ['Delivery', 'VAT', 'Surcharge']:
            if col not in discount_df.columns:
                discount_df[col] = 0
        discount_df['CM'] = discount_df['net_sale'] - discount_df['Calculated_Discount'] - discount_df['Delivery'] - discount_df['VAT'] - discount_df['Surcharge']

    # --- Brandwise Discount Analysis ---
    brandwise = None
    if 'Brand' in discount_df.columns:
        brandwise = discount_df.groupby('Brand').agg(
            Orders=('OrderID', 'count'),
            Total_Discount=('Calculated_Discount', 'sum'),
            Avg_Discount=('Calculated_Discount', 'mean'),
            Total_CM=('CM', 'sum'),
            Avg_CM=('CM', 'mean'),
            Total_Net_Sale=('net_sale', 'sum')
        ).sort_values('Total_Discount', ascending=False)
        brandwise = sanitize_for_streamlit(brandwise.reset_index())

    # --- Addresswise Discount Analysis ---
    locationwise = None
    if 'Address' in discount_df.columns:
        locationwise = discount_df.groupby('Address').agg(
            Orders=('OrderID', 'count'),
            Total_Discount=('Calculated_Discount', 'sum'),
            Avg_Discount=('Calculated_Discount', 'mean'),
            Total_CM=('CM', 'sum'),
            Avg_CM=('CM', 'mean'),
            Total_Net_Sale=('net_sale', 'sum')
        ).sort_values('Total_Discount', ascending=False)
        locationwise = sanitize_for_streamlit(locationwise.reset_index())

    # --- Brand-Address Pairwise Discount Analysis ---
    pairwise = None
    if 'Brand' in discount_df.columns and 'Address' in discount_df.columns:
        pairwise = discount_df.groupby(['Brand', 'Address']).agg(
            Orders=('OrderID', 'count'),
            Total_Discount=('Calculated_Discount', 'sum'),
            Avg_Discount=('Calculated_Discount', 'mean'),
            Total_CM=('CM', 'sum'),
            Avg_CM=('CM', 'mean'),
            Total_Net_Sale=('net_sale', 'sum')
        ).sort_values('Total_Discount', ascending=False)
        pairwise = sanitize_for_streamlit(pairwise.reset_index())

    # --- Discount Range Analysis (existing) ---
    discount_analysis = discount_df.groupby('Discount_Range').agg({
        'net_sale': ['sum', 'mean'],
        'OrderID': 'count',
        'Calculated_Discount': 'mean',
        'CM': ['sum', 'mean']
    }).round(2)
    discount_analysis.columns = ['Total_Net_Sales', 'Average_Order_Value', 'Number_of_Orders', 'Average_Discount_Amount', 'Total_CM', 'Average_CM']
    discount_analysis = discount_analysis.reset_index()

    # --- Visualization (existing) ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Discount Distribution by Amount', 'Revenue by Discount Range',
                       'Brandwise Discount CM', 'Discount Impact on Order Value'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    # Discount distribution
    fig.add_trace(
        go.Histogram(x=discount_df['Calculated_Discount'], name='Discount Distribution',
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
    # Brandwise CM (if available)
    if brandwise is not None:
        fig.add_trace(
            go.Bar(x=brandwise.index, y=brandwise['Total_CM'],
                   name='Brandwise CM', marker_color='#2ca02c'),
            row=2, col=1
        )
    # Discount impact
    fig.add_trace(
        go.Scatter(x=discount_df['Calculated_Discount'], y=discount_df['net_sale'],
                  mode='markers', name='Discount vs Revenue',
                  marker=dict(size=6, color='#d62728', opacity=0.6)),
        row=2, col=2
    )
    fig.update_layout(height=600, showlegend=True, title_text="Discount Performance Analysis")
    return fig, brandwise, locationwise, pairwise, discount_analysis

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
            'Calculated_Discount': 'sum'
        }).reset_index()
        
        daily_sales['Date_ordinal'] = pd.to_datetime(daily_sales['Date']).map(datetime.toordinal)
        daily_sales['DayOfWeek'] = pd.to_datetime(daily_sales['Date']).dt.dayofweek
        daily_sales['Month'] = pd.to_datetime(daily_sales['Date']).dt.month
        
        # Features for prediction
        features = ['Date_ordinal', 'DayOfWeek', 'Month', 'OrderID', 'Calculated_Discount']
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
            'Calculated_Discount': daily_sales['Calculated_Discount'].mean()
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
    
    # Sales by Address
    if 'Address' in df.columns:
        address_sales = df.groupby('Address')['net_sale'].sum().nlargest(10)
        fig.add_trace(
            go.Bar(x=address_sales.index, y=address_sales.values,
                   name='Address Sales', marker_color='#ff7f0e'),
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

    # Always define df_filtered as a copy of df at the start for robustness
    df_filtered = df.copy()
    # ...existing code before date filtering...

    # ...date filtering and df_filtered assignment...

    # --- Deep Dive Discount Performance Insights ---
    # (Moved to a dedicated function, call only in Discount Performance tab)
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
                    'Calculated_Discount': 'sum',
                    'OrderID': 'count'
                }).reset_index()
                
                # Calculate trends
                avg_net_sale = daily_stats['net_sale'].mean()
                avg_discount = daily_stats['Calculated_Discount'].mean()
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
    if 'net_sale' in df.columns and 'Calculated_Discount' in df.columns and 'GrossPrice' in df.columns:
        # Ensure Discount_Percentage is calculated from GrossPrice using Calculated_Discount
        df['Discount_Percentage'] = (df['Calculated_Discount'] / df['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
    
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
        st.warning("⚠️ Date column not found. Using all data.")
    
    # Apply exact sync filter by default (per-date latest timestamp to the minute)
    sync_info = None
    sync_meta = {}
    if 'ReceivedAt' in df_filtered.columns and len(df_filtered) > 0 and 'Date' in df_filtered.columns:
        filtered_parts = []
        unique_dates = sorted(df_filtered['Date'].dt.date.unique())
        if len(unique_dates) == 2:
            # Always treat previous day as the second latest day (not min, but second max)
            latest_date = unique_dates[-1]
            prev_date = unique_dates[-2]
            part_latest = df_filtered[df_filtered['Date'].dt.date == latest_date].copy()
            part_prev = df_filtered[df_filtered['Date'].dt.date == prev_date].copy()
            if not part_latest.empty:
                max_ts = part_latest['ReceivedAt'].max()
                # Filter latest date up to its own latest timestamp (exact hour, minute, second)
                filtered_latest = part_latest[part_latest['ReceivedAt'] <= max_ts].copy()
                filtered_latest['Hour'] = filtered_latest['ReceivedAt'].dt.hour
                filtered_latest['Minute'] = filtered_latest['ReceivedAt'].dt.minute
                filtered_parts.append(filtered_latest)
                sync_meta[latest_date] = {
                    'latest_timestamp': max_ts,
                    'latest_hour': max_ts.hour,
                    'latest_minute': max_ts.minute,
                    'latest_second': max_ts.second
                }
                if not part_prev.empty:
                    # For previous day (second latest), filter up to the exact hour, minute, and second of latest day's max timestamp
                    prev_cutoff = pd.Timestamp.combine(prev_date, max_ts.time())
                    filtered_prev = part_prev[part_prev['ReceivedAt'] <= prev_cutoff].copy()
                    # If previous day has more records at the cutoff timestamp than latest day, only include up to the same count
                    n_latest_at_cutoff = (filtered_latest['ReceivedAt'] == max_ts).sum()
                    prev_at_cutoff_mask = filtered_prev['ReceivedAt'] == prev_cutoff
                    n_prev_at_cutoff = prev_at_cutoff_mask.sum()
                    if n_prev_at_cutoff > n_latest_at_cutoff:
                        # Only keep up to n_latest_at_cutoff records at the cutoff timestamp
                        idx_to_keep = filtered_prev[prev_at_cutoff_mask].index[:n_latest_at_cutoff]
                        # Keep all records before cutoff, and only up to n_latest_at_cutoff at cutoff
                        filtered_prev = pd.concat([
                            filtered_prev[filtered_prev['ReceivedAt'] < prev_cutoff],
                            filtered_prev.loc[idx_to_keep]
                        ], ignore_index=True)
                    filtered_prev['Hour'] = filtered_prev['ReceivedAt'].dt.hour
                    filtered_prev['Minute'] = filtered_prev['ReceivedAt'].dt.minute
                    filtered_parts.append(filtered_prev)
                    sync_meta[prev_date] = {
                        'latest_timestamp': prev_cutoff,
                        'latest_hour': max_ts.hour,
                        'latest_minute': max_ts.minute,
                        'latest_second': max_ts.second
                    }
            if filtered_parts:
                df_filtered = pd.concat(filtered_parts, ignore_index=True)
                sync_info = sync_meta
            else:
                sync_info = None
        else:
            # Fallback: for single date, use its own latest timestamp
            for d in unique_dates:
                part = df_filtered[df_filtered['Date'].dt.date == d].copy()
                if not part.empty:
                    max_ts = part['ReceivedAt'].max()
                    max_ts_minute = max_ts.replace(second=0, microsecond=0)
                    filtered = part[part['ReceivedAt'] <= max_ts_minute].copy()
                    filtered['Hour'] = filtered['ReceivedAt'].dt.hour
                    filtered['Minute'] = filtered['ReceivedAt'].dt.minute
                    filtered_parts.append(filtered)
                    sync_meta[d] = {
                        'latest_timestamp': max_ts_minute,
                        'latest_hour': max_ts_minute.hour,
                        'latest_minute': max_ts_minute.minute
                    }
            if filtered_parts:
                df_filtered = pd.concat(filtered_parts, ignore_index=True)
                sync_info = sync_meta
            else:
                sync_info = None
    
    # Sidebar controls for pivot table configuration
    st.subheader("🔧 Pivot Table Configuration")
    # Show per-date sync info
    if sync_info:
        st.markdown("### ⏰ Exact Sync Info")
        for d, meta in sync_info.items():
            st.info(f"🕐 **{d}**: Up to {meta['latest_timestamp'].strftime('%H:%M')} (auto-synchronized)")
    # Optional hour filter (user must opt-in)
    selected_max_hour = None
    if sync_info and st.checkbox("Apply Hour Filter (override exact sync)?", value=False, help="Enable to filter by hour instead of exact sync time."):
        # User wants to apply hour filter
        available_hours = sorted(df_filtered['Hour'].unique()) if 'Hour' in df_filtered.columns else []
        if available_hours:
            max_hour = max([meta['latest_hour'] for meta in sync_info.values()])
            selected_max_hour = st.selectbox(
                "📊 Filter to Hour:",
                options=list(range(max_hour + 1)),
                index=max_hour,
                format_func=lambda x: f"Up to {x:02d}:xx ({x+1} hours)",
                help=f"Select maximum hour to include. This will override the exact sync filter."
            )
            # Apply hour filter to the data (all dates)
            df_filtered = df_filtered[df_filtered['Hour'] <= selected_max_hour].copy()
            st.success(f"✅ **Hour Filter Applied**: {len(df_filtered):,} records (Hours 0-{selected_max_hour})")
        else:
            st.warning("⚠️ No hour data available for filtering")
            selected_max_hour = None
    
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
    if 'Address' in df_filtered.columns:
        categorical_columns.append('Address')
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
                'Calculated_Discount': 'sum',
                'GrossPrice': 'sum',
                'OrderID': 'count'
            }).reset_index()
            
            # Check if pivot_data has any rows
            if len(pivot_data) == 0:
                st.warning("⚠️ No data found for Channel and Date grouping. Please check your data.")
                return
            
            # Calculate Discount Percentage (from GrossPrice, not net_sale)
            pivot_data['Discount_Percentage'] = (pivot_data['Calculated_Discount'] / pivot_data['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
            
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
                values='Calculated_Discount',
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
                # Net Sale
                net_sale = pivot_net_sale[date] if date in pivot_net_sale.columns else 0
                combined_data[(date_str, 'Net Sale')] = net_sale
                # Discount
                discount = pivot_discount[date] if date in pivot_discount.columns else 0
                combined_data[(date_str, 'Discount')] = discount
                # Discount %
                discount_pct = pivot_discount_pct[date] if date in pivot_discount_pct.columns else 0
                combined_data[(date_str, 'Discount %')] = discount_pct
                # Contribution Margin (CM): CM = Net Sale - Food Cost - Commission - Discount
                # Food Cost = Net Sale * 20.6%, Commission = Net Sale * 24%
                food_cost = net_sale * 0.206
                commission = net_sale * 0.24
                cm = net_sale - food_cost - commission - discount
                combined_data[(date_str, 'CM')] = cm
            
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
                food_cost_total = net_sale_total * 0.206
                commission_total = net_sale_total * 0.24
                cm_total = net_sale_total - food_cost_total - commission_total - discount_total
                totals_row[(date_str, 'Net Sale')] = net_sale_total
                totals_row[(date_str, 'Discount')] = discount_total
                totals_row[(date_str, 'Discount %')] = discount_pct_avg
                totals_row[(date_str, 'CM')] = cm_total
            
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
                combined_pivot[(f'Δ {second_date} vs {first_date}', 'CM Δ')] = (
                    combined_pivot[(second_date, 'CM')] - combined_pivot[(first_date, 'CM')]
                )
                # Recreate columns to maintain proper order (chronological order)
                new_columns = []
                for date in sorted_dates:
                    date_str = str(date)
                    new_columns.extend([
                        (date_str, 'Net Sale'),
                        (date_str, 'Discount'),
                        (date_str, 'Discount %'),
                        (date_str, 'CM')
                    ])
                # Add difference columns at the end
                new_columns.extend([
                    (f'Δ {second_date} vs {first_date}', 'Net Sale Δ'),
                    (f'Δ {second_date} vs {first_date}', 'Discount Δ'),
                    (f'Δ {second_date} vs {first_date}', 'Discount % Δ'),
                    (f'Δ {second_date} vs {first_date}', 'CM Δ')
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
                elif 'CM' in col[1]:
                    formatted_combined[col] = formatted_combined[col].round(2)
                else:
                    formatted_combined[col] = formatted_combined[col].round(2)
            
            # Create custom styling for difference columns
            def style_differences(val):
                if pd.isna(val):
                    return ''
                # For discount differences: reduction (negative) should be green, increase (positive) should be red
                # For net sale and CM differences: increase (positive) should be green, decrease (negative) should be red
                if val > 0:
                    return 'background-color: #d4edda; color: #155724'  # Green for positive (good for net sales/CM)
                elif val < 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Red for negative (bad for net sales/CM)
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
                col: '{:.2f}' if 'Discount %' not in col[1] and 'CM' not in col[1] else ('{:.1f}%' if 'Discount %' in col[1] else '{:.2f}')
                for col in formatted_combined.columns
            })
            
            # Apply difference column styling if they exist
            if len(dates) == 2:
                diff_cols = [col for col in formatted_combined.columns if 'Δ' in col[0]]
                for col in diff_cols:
                    if 'Discount Δ' in col[1] or 'Discount % Δ' in col[1]:
                        # For discount metrics: reduction (negative) = green, increase (positive) = red
                        styled_df = styled_df.applymap(style_discount_differences, subset=[col])
                    elif 'CM Δ' in col[1]:
                        # For CM: increase (positive) = green, decrease (negative) = red
                        styled_df = styled_df.applymap(style_differences, subset=[col])
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
                    st.plotly_chart(fig_heatmap, use_container_width=True, key="net_sale_heatmap")
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
                    st.plotly_chart(fig_heatmap_pct, use_container_width=True, key="discount_pct_heatmap")
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
                            'Calculated_Discount': 'sum',
                            'GrossPrice': 'sum',
                            'OrderID': 'count'
                        }).reset_index()
                        
                        # Calculate Discount Percentage
                        analysis_data['Discount_Percentage'] = (analysis_data['Calculated_Discount'] / analysis_data['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
                        
                        # Create pivot for analysis
                        analysis_pivot_net = analysis_data.pivot_table(
                            values='net_sale',
                            index=selected_dimension,
                            columns='Date',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        analysis_pivot_discount = analysis_data.pivot_table(
                            values='Calculated_Discount',
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
                                    st.dataframe(sanitize_for_streamlit(display_top), use_container_width=True)
                            
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
                                    st.dataframe(sanitize_for_streamlit(display_bottom), use_container_width=True)
                            
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
                            'Calculated_Discount': 'sum',
                            'GrossPrice': 'sum',
                            'OrderID': 'count'
                        }).reset_index()
                        
                        # Calculate Discount Percentage
                        combined_analysis['Discount_Percentage'] = (combined_analysis['Calculated_Discount'] / combined_analysis['GrossPrice'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
                        
                        # Create pivot for combined dimensions
                        combined_pivot_net = combined_analysis.pivot_table(
                            values='net_sale',
                            index=available_dimensions,
                            columns='Date',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        combined_pivot_discount = combined_analysis.pivot_table(
                            values='Calculated_Discount',
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
                                        label=" Worst Decline",
                                        value=worst_decline['Combined_Dimensions'][:50] + "...",  # Truncate for display
                                        delta=f"{worst_decline['Net_Sale_Diff']:,.0f}"
                                    )
                                else:
                                    st.metric(label=" Worst Decline", value="None", delta="All positive!")
                            
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
                        # Robustly find the Net Sale Δ column (second element == 'Net Sale Δ')
                        net_sale_diff_col = None
                        for col in diff_cols:
                            if isinstance(col, tuple) and col[1] == 'Net Sale Δ':
                                net_sale_diff_col = col
                                break
                        discount_diff_col = next((col for col in diff_cols if isinstance(col, tuple) and col[1] == 'Discount Δ'), None)
                        discount_pct_diff_col = next((col for col in diff_cols if isinstance(col, tuple) and col[1] == 'Discount % Δ'), None)
                        
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
            st.dataframe(sanitize_for_streamlit(pivot_table.round(2)), use_container_width=True)
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
        # BigQuery debug UI removed as per user request

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
    # --- Recommendations for further analysis and decision making ---
    with st.expander("💡 Dashboard Recommendations & Advanced Analysis Features"):
        st.markdown("""
        **To make your dashboard even more powerful for data analysis and decision making, consider these features:**
        - **Advanced Filtering & Segmentation:**
            - Multi-select filters for Location, Payment Method, Customer Segment, etc.
            - Date granularity toggle (daily, weekly, monthly).
            - Custom filter builder for complex queries.
        - **Enhanced Visualizations:**
            - Time series decomposition (trend, seasonality, anomalies).
            - Cohort analysis for customer retention.
            - Funnel and conversion analysis.
            - Heatmaps for sales by hour/day.
        - **Comparative & Drilldown Analysis:**
            - Drilldown charts (click to see details by brand, product, etc.).
            - Side-by-side period comparison for any two custom date ranges.
            - Contribution analysis to identify top drivers of growth/decline.
        - **Predictive & Prescriptive Analytics:**
            - Forecasting with confidence intervals.
            - What-if analysis (simulate impact of changes).
            - Churn prediction for customers.
        - **Data Quality & Integrity:**
            - Data freshness indicator (already added).
            - Outlier detection and missing data heatmaps.
        - **User Experience & Sharing:**
            - Customizable dashboards (save favorite views).
            - Export to Excel/PowerPoint.
            - Scheduled email reports.
        - **Actionable Insights:**
            - Automated plain-English summaries of key trends.
            - Alerting when KPIs cross thresholds.
        - **Performance & Scalability:**
            - Asynchronous data loading and caching for large datasets.
        """)
    st.title("Growth Team Dashboard")
    # Data Freshness Indicator
    data_load_time = st.session_state.get('data_load_time', None)
    if data_load_time is not None:
        st.info(f"🕒 Data last loaded: {data_load_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("""
        Welcome to the Sales Analytics Dashboard. Use the sidebar to filter or change the data source.
    """)
    

    # Sidebar controls
    with st.sidebar:
        st.subheader("Data Source Settings")
        data_source = "MySQL"  # Default data source (MySQL only)
        st.info(f"Using {data_source} as the data source.")

        # Date range filter (Default: Month-To-Date)
        st.subheader("Date Range Filter")
        today = datetime.now().date()
        default_start = today.replace(day=1)  # Start of current month
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

        # Calculate days_back for backward compatibility (for MySQL/BigQuery loaders)
        days_back = (today - start_date).days + 1
        refresh_button = st.button("Refresh Data")

        # Add a progress indicator in sidebar
        if 'loading_state' not in st.session_state:
            st.session_state.loading_state = "ready"

        # Show a loading indicator
        if st.session_state.loading_state == "loading":
            with st.spinner("Loading data..."):
                st.progress(0.75, "Processing data")

        # Show tip for data loading
        st.info("💡 **Tip**: Use fewer days for quicker loading with large datasets.")
    
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
                # Load data from MySQL database
                df = load_data_from_mysql(days_back)
                if not df.empty:
                    df = add_essential_calculated_columns(df)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.data_load_time = datetime.now()
                    st.success(f"✅ Loaded {len(df):,} records from {data_source}.")
                else:
                    st.warning(f"⚠️ No data loaded from {data_source}. Please check your connection settings.")
            st.session_state.loading_state = "ready"
        except Exception as e:
            st.session_state.loading_state = "ready"
            st.error(f"Error loading data: {str(e)}")
    
    # Get the dataframe from session state
    df = st.session_state.df

    # --- Date filter for in-memory data (UI only, does not reload from DB) ---
    if df is not None and not df.empty:
        # Only filter if Date column exists
        if 'Date' in df.columns:
            # Ensure Date is datetime64[ns] for comparison
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            # Convert start_date and end_date to pandas.Timestamp for valid comparison
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            filtered_df = df[(df['Date'] >= start_ts) & (df['Date'] <= end_ts)]
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
            "📅 Month-to-Date (MTD) Analysis",
            "👥 Customer Behavior",
            "🎫 Discount Performance",
            "⚖️ Comparative Analysis",
            "🧪 Sandbox",
            "👤 Customer Segmentation",
            "🎮 Shady's Command Center",
            "📅 Period Comparison",
            "📋 Data Overview",
            "🧹 Data Quality"
        ])
        # 5. Sandbox Tab (User-Friendly & Interactive)
        with tabs[4]:
            st.header("🧪 Sandbox: Interactive Performance Explorer")
            st.markdown("""
            <span style='font-size:1.1em;'>
            <b>Explore sales, discount, and profitability by any combination of <span style='color:#1f77b4;'>Branch</span>, <span style='color:#ff7f0e;'>Brand</span>, <span style='color:#2ca02c;'>Channel</span>, or <span style='color:#d62728;'>Location</span>.</b><br>
            <ul>
            <li>🧩 <b>Step 1:</b> <b>Select dimensions</b> to group by (e.g., Brand, Channel, Location, Branch).</li>
            <li>📊 <b>Step 2:</b> <b>Choose metrics</b> to analyze (Net Sales, Discount, CM, etc.).</li>
            <li>📈 <b>Step 3:</b> <b>Visualize</b> and <b>download</b> results for further analysis.</li>
            </ul>
            </span>
            """, unsafe_allow_html=True)
            # Choose dimensions
            available_dims = []
            if 'Brand' in filtered_df.columns:
                available_dims.append('Brand')
            if 'Channel' in filtered_df.columns:
                available_dims.append('Channel')
            if 'Location' in filtered_df.columns:
                available_dims.append('Location')
            if 'Branch' in filtered_df.columns:
                available_dims.append('Branch')
            if not available_dims:
                st.warning("No Brand, Channel, Location, or Branch columns found in data.")
            else:
                with st.container():
                    st.markdown("<b>🧩 Select Dimensions:</b>", unsafe_allow_html=True)
                    selected_dims = st.multiselect(
                        "Group by (choose 1-3):",
                        options=available_dims,
                        default=available_dims[:1],
                        help="Choose one or more dimensions to analyze."
                    )
                # Select metrics
                metric_options = [
                    ('net_sale', 'Net Sales'),
                    ('OrderID', 'Orders'),
                    ('Calculated_Discount', 'Discount'),
                    ('CM', 'Contribution Margin'),
                    ('Profit_Margin', 'Profit Margin'),
                ]
                available_metrics = [(col, label) for col, label in metric_options if col in filtered_df.columns]
                with st.container():
                    st.markdown("<b>📊 Select Metrics:</b>", unsafe_allow_html=True)
                    selected_metrics = st.multiselect(
                        "Metrics to display:",
                        options=[label for col, label in available_metrics],
                        default=[label for col, label in available_metrics[:3]],
                        help="Choose metrics to summarize."
                    )
                # Map labels back to column names
                metric_map = {label: col for col, label in available_metrics}
                selected_metric_cols = [metric_map[label] for label in selected_metrics]
                # Optional: Add filter for top N
                with st.expander("🔎 Advanced: Filter & Sort"):
                    top_n = st.slider("Show Top N (by first metric)", min_value=5, max_value=100, value=20, step=1)
                    sort_desc = st.checkbox("Sort descending (highest first)", value=True)
                if selected_dims and selected_metric_cols:
                    # Group and summarize
                    agg_dict = {col: 'sum' if col != 'Profit_Margin' else 'mean' for col in selected_metric_cols}
                    summary = filtered_df.groupby(selected_dims).agg(agg_dict).reset_index()
                    # Sort and filter top N
                    if selected_metric_cols:
                        sort_col = selected_metric_cols[0]
                        summary = summary.sort_values(sort_col, ascending=not sort_desc).head(top_n)
                    st.subheader("Summary Table")
                    st.dataframe(sanitize_for_streamlit(summary.round(2)), use_container_width=True, hide_index=True)
                    # Download option
                    csv = summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv,
                        file_name=f"sandbox_summary_{'_'.join(selected_dims)}.csv",
                        mime="text/csv"
                    )
                    # Visualization
                    st.markdown("<b>📈 Visualizations:</b>", unsafe_allow_html=True)
                    import plotly.express as px
                    if len(selected_dims) == 1 and len(selected_metric_cols) >= 1:
                        chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True, key="sandbox_chart_type1")
                        for col in selected_metric_cols:
                            if chart_type == "Bar":
                                fig = px.bar(summary, x=selected_dims[0], y=col, title=f"{col} by {selected_dims[0]}")
                            else:
                                fig = px.pie(summary, names=selected_dims[0], values=col, title=f"{col} by {selected_dims[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                    elif len(selected_dims) == 2 and len(selected_metric_cols) >= 1:
                        chart_type = st.radio("Chart Type", ["Grouped Bar", "Heatmap"], horizontal=True, key="sandbox_chart_type2")
                        for col in selected_metric_cols:
                            if chart_type == "Grouped Bar":
                                fig = px.bar(summary, x=selected_dims[0], y=col, color=selected_dims[1], barmode='group', title=f"{col} by {selected_dims[0]} and {selected_dims[1]}")
                            else:
                                fig = px.density_heatmap(summary, x=selected_dims[0], y=selected_dims[1], z=col, color_continuous_scale='Viridis', title=f"{col} Heatmap: {selected_dims[0]} vs {selected_dims[1]}")
                            st.plotly_chart(fig, use_container_width=True)
                    elif len(selected_dims) == 3 and len(selected_metric_cols) >= 1:
                        st.info("For 3+ dimensions, download the summary for deeper analysis or use external BI tools.")
                        # Optionally, show a sample table
                        st.dataframe(sanitize_for_streamlit(summary.head(10).round(2)), use_container_width=True, hide_index=True)
                else:
                    st.info("Select at least one dimension and one metric to view results.")
            st.write("### Data Preview")
            st.dataframe(sanitize_for_streamlit(filtered_df.head(20)), use_container_width=True)

        # 1. Month-to-Date (MTD) Analysis Tab (replaces Sales Trends)

        with tabs[0]:
            # Month-to-Date (MTD) Analysis Tab
            st.header("📅 Month-to-Date (MTD) Analysis")
            mtd_df = filtered_df.copy()
            # Show summary metrics
            metrics = calculate_metrics(mtd_df)
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Orders", f"{metrics['total_orders']:,}")
                col2.metric("Total Revenue", f"{metrics['total_revenue']:,.2f}")
                col3.metric("Avg Order Value", f"{metrics['avg_order_value']:,.2f}")
                col4.metric("Total Customers", f"{metrics['total_customers']:,}")
                col1.metric("Total Discount", f"{metrics['total_discount']:,.2f}")
                col2.metric("Discount Rate (%)", f"{metrics['discount_rate']:.1f}%")
                col3.metric("Avg Delivery Fee", f"{metrics['avg_delivery_fee']:,.2f}")
                col4.metric("Total Tips", f"{metrics['total_tips']:,.2f}")
            else:
                st.info("No data available for MTD metrics.")
            # Show sales trends chart
            st.plotly_chart(create_sales_trends_chart(mtd_df), use_container_width=True)
            # Show sample data
            st.subheader("Sample Data (MTD)")
            st.dataframe(sanitize_for_streamlit(mtd_df.head(20)), use_container_width=True)

        with tabs[1]:
            # Customer Behavior Tab
            st.header("👥 Customer Behavior Analysis")
            fig, customer_stats = create_customer_behavior_analysis(filtered_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            if customer_stats is not None and not customer_stats.empty:
                st.subheader("Customer Stats Table")
                st.dataframe(sanitize_for_streamlit(customer_stats.head(20)), use_container_width=True)
                # Enhanced insights for customer behavior
                st.markdown("### 💡 Customer Behavior Insights")
                repeat_rate = (customer_stats['Order_Count'] > 1).mean() * 100
                avg_lifetime = customer_stats['Customer_Lifetime_Days'].mean() if 'Customer_Lifetime_Days' in customer_stats.columns else None
                top_segment = customer_stats['Segment'].value_counts().idxmax() if 'Segment' in customer_stats.columns else None
                st.write(f"• **Repeat Customer Rate:** {repeat_rate:.1f}% of customers have placed more than one order.")
                if avg_lifetime is not None:
                    st.write(f"• **Average Customer Lifetime:** {avg_lifetime:.1f} days")
                if top_segment is not None:
                    st.write(f"• **Largest Segment:** {top_segment}")
                st.write("• **Actionable Tip:** Target high-value and repeat customers with loyalty offers. Monitor low-value segments for churn risk.")
            else:
                st.info("No customer data available for analysis.")

        with tabs[2]:
            # --- Location & Brand Discount Campaign Performance ---
            st.subheader("📍 Location & Brand Discount Campaign Performance")
            if 'Location' in filtered_df.columns and 'Brand' in filtered_df.columns and 'Calculated_Discount' in filtered_df.columns and 'net_sale' in filtered_df.columns:
                loc_brand = filtered_df.groupby(['Location', 'Brand']).agg(
                    Total_Orders=('OrderID', 'count'),
                    Total_Discount=('Calculated_Discount', 'sum'),
                    Avg_Discount=('Calculated_Discount', 'mean'),
                    Total_Sales=('net_sale', 'sum'),
                    Avg_Sale=('net_sale', 'mean')
                ).sort_values('Total_Discount', ascending=False)
                # Calculate Discount Rate and Profitability
                loc_brand['Discount_Rate_%'] = (loc_brand['Total_Discount'] / (loc_brand['Total_Sales'] + loc_brand['Total_Discount']) * 100).round(2)
                # Assume a simple profitability: Net Sale - Discount
                loc_brand['Profit'] = loc_brand['Total_Sales'] - loc_brand['Total_Discount']
                # Locations performing well: high sales, positive profit, reasonable discount rate
                good_mask = (loc_brand['Profit'] > 0) & (loc_brand['Discount_Rate_%'] < 35) & (loc_brand['Total_Sales'] > 0)
                bad_mask = (loc_brand['Profit'] < 0) | (loc_brand['Discount_Rate_%'] > 50)
                st.markdown("### 🟢 Locations Performing Well with Discount Campaigns")
                good_df = loc_brand[good_mask].sort_values('Profit', ascending=False).head(10)
                if not good_df.empty:
                    st.dataframe(good_df[['Total_Orders','Total_Sales','Total_Discount','Discount_Rate_%','Profit']].reset_index(), use_container_width=True)
                else:
                    st.info("No locations found with strong positive performance on discounts.")
                st.markdown("### 🔴 Locations Where Discounts Are Negatively Impacting Sales")
                bad_df = loc_brand[bad_mask].sort_values('Profit').head(10)
                if not bad_df.empty:
                    st.dataframe(bad_df[['Total_Orders','Total_Sales','Total_Discount','Discount_Rate_%','Profit']].reset_index(), use_container_width=True)
                else:
                    st.info("No locations found with negative impact from discounts.")
            # Discount Performance Tab
            st.header("🎫 Discount Performance Analysis")
            fig, brandwise, locationwise, pairwise, discount_analysis = create_discount_performance_analysis(filtered_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### 💡 Discount Performance Deep Dive & Insights")
                # Call the deep dive function here
                deep_dive_discount_performance_insights(filtered_df)
                # --- Brand-Location Pairwise Table (Organized) ---
                if pairwise is not None and not pairwise.empty:
                    st.subheader("Brand-Location Discount & Contribution Margin Table")
                    display_cols = ['Orders', 'Total_Discount', 'Avg_Discount', 'Total_CM', 'Avg_CM', 'Total_Net_Sale']
                    st.dataframe(sanitize_for_streamlit(pairwise[display_cols].sort_values('Total_CM', ascending=False).head(30)), use_container_width=True)
                    # Recommendations by Brand@Location
                    st.markdown("**Brand@Location Recommendations:**")
                    profitable = []
                    negative = []
                    for idx, row in pairwise.iterrows():
                        brand, location = idx
                        if row['Total_Discount'] > 0:
                            if row['Total_CM'] < 0:
                                negative.append(f"🔴 {brand} @ {location} (CM: {row['Total_CM']:.2f})")
                            elif row['Total_CM'] > 0 and row['Avg_CM'] > 5:
                                profitable.append(f"🟢 {brand} @ {location} (CM: {row['Total_CM']:.2f})")
                    if profitable:
                        st.write("**Profitable on Discount (Brand@Location):**")
                        for rec in profitable[:10]:
                            st.write(rec)
                    if negative:
                        st.write("**Negative Impact (Brand@Location):**")
                        for rec in negative[:10]:
                            st.write(rec)
                # --- Brandwise Table ---
                if brandwise is not None and not brandwise.empty:
                    st.subheader("Brandwise Discount & Contribution Margin")
                    st.dataframe(sanitize_for_streamlit(brandwise.round(2).head(20)), use_container_width=True)
                # --- Locationwise Table ---
                if locationwise is not None and not locationwise.empty:
                    st.subheader("Locationwise Discount & Contribution Margin")
                    st.dataframe(sanitize_for_streamlit(locationwise.round(2).head(20)), use_container_width=True)
                # --- Discount Range Table ---
                if discount_analysis is not None and not discount_analysis.empty:
                    st.subheader("Discount Range Performance (with CM)")
                    st.dataframe(sanitize_for_streamlit(discount_analysis.round(2)), use_container_width=True)
                    st.markdown("**Discount Range Insights:**")
                    for _, row in discount_analysis.iterrows():
                        if row['Total_CM'] < 0:
                            st.write(f"🔴 Range {row['Discount_Range']}: Negative CM. Avoid or revise.")
                        elif row['Average_CM'] < 2:
                            st.write(f"🟠 Range {row['Discount_Range']}: Marginal CM. Monitor closely.")
                        elif row['Total_CM'] > 0 and row['Average_CM'] > 5:
                            st.write(f"🟢 Range {row['Discount_Range']}: Profitable. Consider scaling.")
                st.markdown("---")
                st.markdown("**General Recommendations:**")
                st.write("• Launch or scale discounts where CM is strongly positive and sales uplift is clear.")
                st.write("• Discontinue or revise discounts with negative CM, especially if sales volume is not compensating.")
                st.write("• Monitor locations/brands/pairs with marginal CM for further review.")
                st.write("• Use A/B testing to validate discount effectiveness before scaling.")
            else:
                st.info("No discount data available for analysis.")

        with tabs[3]:
            # Comparative Analysis Tab
            st.header("⚖️ Comparative Analysis")
            fig = create_comparison_analysis(filtered_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for comparative analysis.")

        # 6. Customer Segmentation Tab
        with tabs[5]:
            # Customer Segmentation Tab
            st.header("👤 Customer Segmentation")
            if 'Customer_Value_Tier' in filtered_df.columns:
                seg_counts = filtered_df['Customer_Value_Tier'].value_counts().sort_index()
                st.bar_chart(seg_counts)
                st.dataframe(filtered_df[['CustomerName', 'Customer_Value_Tier']].drop_duplicates().head(20), use_container_width=True)
                # Deep Dive: Segment Metrics
                st.subheader("🔍 Segment Metrics & Deep Dive")
                segment_metrics = filtered_df.groupby('Customer_Value_Tier').agg(
                    Total_Customers=('CustomerName', 'nunique'),
                    Total_Orders=('OrderID', 'count'),
                    Total_Spent=('net_sale', 'sum'),
                    Avg_Order_Value=('net_sale', 'mean'),
                    Total_Discount=('Calculated_Discount', 'sum'),
                    Avg_Discount=('Calculated_Discount', 'mean'),
                    Profit_Margin=('Profit_Margin', 'mean'),
                    Repeat_Customers=('CustomerName', lambda x: (x.value_counts() > 1).sum())
                ).round(2)
                st.dataframe(segment_metrics, use_container_width=True)
                # Visualize segment profitability
                st.write("#### Segment Profitability")
                st.bar_chart(segment_metrics['Profit_Margin'])
                # Visualize repeat customer rate
                st.write("#### Repeat Customer Rate by Segment")
                segment_metrics['Repeat_Rate_%'] = (segment_metrics['Repeat_Customers'] / segment_metrics['Total_Customers'] * 100).round(1)
                st.bar_chart(segment_metrics['Repeat_Rate_%'])
                # Insights
                st.markdown("### 💡 Customer Segmentation Insights & Actions")
                top_tier = seg_counts.idxmax() if not seg_counts.empty else None
                st.write(f"• **Largest Value Tier:** {top_tier}")
                profitable_tiers = segment_metrics[segment_metrics['Profit_Margin'] > 10].index.tolist()
                if profitable_tiers:
                    st.write(f"🟢 Profitable Segments: {', '.join([str(t) for t in profitable_tiers])}")
                low_margin_tiers = segment_metrics[segment_metrics['Profit_Margin'] < 5].index.tolist()
                if low_margin_tiers:
                    st.write(f"🔴 Low-Margin Segments: {', '.join([str(t) for t in low_margin_tiers])} (review discounting or cost structure)")
                # Highlight segments with high repeat rate
                high_repeat = segment_metrics[segment_metrics['Repeat_Rate_%'] > 50].index.tolist()
                if high_repeat:
                    st.write(f"🔁 High Repeat Rate Segments: {', '.join([str(t) for t in high_repeat])}")
                st.write("• Focus marketing on Gold/Platinum tiers for upsell, and nurture Bronze/Silver for retention.")
                st.write("• Consider targeted offers for low-margin or low-repeat segments to improve profitability and loyalty.")

                # --- Targeted Marketing Suggestions ---
                st.markdown("### 🎯 Targeted Marketing Suggestions by Segment")
                for tier in segment_metrics.index:
                    metrics = segment_metrics.loc[tier]
                    repeat_rate = metrics['Repeat_Rate_%']
                    profit_margin = metrics['Profit_Margin']
                    avg_order_value = metrics['Avg_Order_Value']
                    total_customers = metrics['Total_Customers']
                    st.markdown(f"**{tier}**")
                    suggestions = []
                    # High value, high repeat
                    if profit_margin > 10 and repeat_rate > 50:
                        suggestions.append("🟢 *Upsell premium products, offer exclusive loyalty rewards, and early access to new launches.*")
                        suggestions.append("💬 *Invite to VIP events or referral programs to leverage advocacy.*")
                    # High value, low repeat
                    elif profit_margin > 10 and repeat_rate <= 50:
                        suggestions.append("🟡 *Send personalized reactivation offers, highlight loyalty program benefits, and use reminders to encourage repeat purchases.*")
                        suggestions.append("💬 *Survey for barriers to repeat purchase and address them directly.*")
                    # Low value, high repeat
                    elif profit_margin <= 10 and repeat_rate > 50:
                        suggestions.append("🟠 *Increase average order value with bundles, cross-sell, or limited-time upgrades. Review discounting to improve margins.*")
                        suggestions.append("💬 *Test price increases or reduce discount depth for this loyal but low-margin group.*")
                    # Low value, low repeat
                    elif profit_margin <= 10 and repeat_rate <= 50:
                        suggestions.append("🔴 *Target with win-back campaigns, special discounts, and customer feedback surveys. Consider reducing discount frequency or value if margins are negative.*")
                        suggestions.append("💬 *Consider pausing costly campaigns and focus on reactivation or churn prevention.*")
                    # General suggestions
                    if avg_order_value > 100:
                        suggestions.append("💡 *Promote high-ticket items and VIP experiences to this segment.*")
                    if total_customers < 20:
                        suggestions.append("📈 *Small segment: test niche offers or gather feedback to grow this tier.*")
                    for s in suggestions:
                        st.write(s)
                st.info("Use these suggestions to tailor campaigns, offers, and communications for each segment. Monitor results and iterate.")
            else:
                st.info("No customer segmentation data available.")
            st.write("### Data Preview")
            st.dataframe(filtered_df.head(20), use_container_width=True)

        # 7. Shady's Command Center Tab (Pivot Table)
        with tabs[6]:
            # Shady's Command Center Tab (Pivot Table + Custom Prediction)
            st.header("🎮 Shady's Command Center")
            # --- Date Filter and Pivot Table Analysis ---
            # This block sets start_date and end_date for the tab
            # ...existing code for date filter and pivot table config...
            # (Find the block where start_date and end_date are set from the tab's date filter)
            # After the date filter and before displaying the pivot table, insert the prediction logic:
            # --- Custom Sales Prediction (Removed) ---
            # If you want to add a new prediction logic, implement here as per your formula
            # ---
            st.header("Pivot Table Analysis")
            try:
                create_pivot_table_analysis(filtered_df)
            except Exception as e:
                st.error(f"Error in pivot table analysis: {e}")
            st.write("### Data Preview")
            st.dataframe(filtered_df.head(20), use_container_width=True)

        # 8. Period Comparison Tab
        with tabs[7]:
            # Period Comparison Tab
            st.header("📅 Period Comparison")
            st.write("### Data Preview for Selected Period")
            st.dataframe(filtered_df.head(20), use_container_width=True)
            st.write("### Data Summary")
            st.dataframe(filtered_df.describe(include='all').T)

            # --- Deep Dive: Period-over-Period Comparison ---
            st.subheader("🔍 Deep Dive: Period-over-Period Comparison")
            if 'Date' in filtered_df.columns:
                # Allow user to select two periods (date ranges)
                unique_dates = sorted(filtered_df['Date'].unique())
                if len(unique_dates) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        period1 = st.selectbox("Select First Period (Date)", unique_dates, index=0)
                    with col2:
                        period2 = st.selectbox("Select Second Period (Date)", unique_dates, index=1 if len(unique_dates) > 1 else 0)
                    df1 = filtered_df[filtered_df['Date'] == period1]
                    df2 = filtered_df[filtered_df['Date'] == period2]
                    # Key metrics to compare
                    metrics = ['net_sale', 'OrderID', 'Calculated_Discount', 'Profit_Margin']
                    summary1 = df1[metrics].sum(numeric_only=True)
                    summary2 = df2[metrics].sum(numeric_only=True)
                    diff = summary2 - summary1
                    comp_df = pd.DataFrame({
                        f'{period1}': summary1,
                        f'{period2}': summary2,
                        'Δ (Second - First)': diff
                    })
                    st.write("#### Key Metrics Comparison")
                    st.dataframe(comp_df.round(2))
                    # Visualize
                    st.write("#### Net Sales Comparison")
                    st.bar_chart(pd.DataFrame({'Net Sale': [summary1['net_sale'], summary2['net_sale']]}, index=[str(period1), str(period2)]))
                    # Insights
                    st.markdown("### 💡 Period Comparison Insights")
                    if diff['net_sale'] > 0:
                        st.write(f"🟢 Net sales increased by {diff['net_sale']:.2f} from {period1} to {period2}.")
                    else:
                        st.write(f"🔴 Net sales decreased by {abs(diff['net_sale']):.2f} from {period1} to {period2}.")
                    if diff['Profit_Margin'] < 0:
                        st.write(f"⚠️ Profit margin dropped by {abs(diff['Profit_Margin']):.2f}. Review cost or discounting.")
                    if diff['Calculated_Discount'] > 0:
                        st.write(f"🔎 Discount outlay increased by {diff['Calculated_Discount']:.2f}. Assess if this drove incremental sales.")
                    st.write("• Use these insights to identify periods of strong or weak performance and investigate drivers.")
                else:
                    st.info("Not enough unique dates for period comparison.")
            else:
                st.info("Date column not found for period comparison.")

        # 8. Data Overview Tab
        with tabs[7]:
            # Data Overview Tab
            st.header("📋 Data Overview")
            st.write("### Data Columns and Types")
            st.dataframe(filtered_df.dtypes.reset_index().rename(columns={0: 'dtype', 'index': 'column'}))
            st.write("### Sample Data")
            st.dataframe(filtered_df.head(20), use_container_width=True)
            st.write("### Data Summary")
            st.dataframe(filtered_df.describe(include='all').T)

        # 9. Data Quality Tab
        with tabs[8]:
            # Data Quality Tab
            st.header("🧹 Data Quality")
            st.write("### Missing Values by Column")
            st.dataframe(filtered_df.isnull().sum().reset_index().rename(columns={0: 'missing', 'index': 'column'}))
            st.write("### Duplicate Rows")
            st.write(f"{filtered_df.duplicated().sum()} duplicate rows found.")
            st.write("### Data Freshness")
            if 'ReceivedAt' in filtered_df.columns:
                st.write(f"Latest ReceivedAt: {filtered_df['ReceivedAt'].max()}")
            else:
                st.info("No ReceivedAt column in data.")
            st.write("### Data Preview")
            st.dataframe(filtered_df.head(20), use_container_width=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log the error but don't show WebSocket errors to users
        if 'WebSocketClosedError' not in str(e) and 'Stream is closed' not in str(e):
            st.error(f"An error occurred: {str(e)}")
        # For WebSocket errors, just log and continue
        pass
