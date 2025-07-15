"""
Timezone Utilities for Sales Dashboard
Handles timezone synchronization between database and application
"""

import mysql.connector
import streamlit as st
import os
from datetime import datetime
import pytz

def check_database_timezone():
    """
    Check current database timezone configuration and provide recommendations.
    """
    try:
        # Get AWS MySQL credentials
        try:
            aws_host = st.secrets["MYSQL_HOST"]
            aws_db = st.secrets["MYSQL_DB"]
            aws_user = st.secrets["MYSQL_USER"]
            aws_pass = st.secrets["MYSQL_PASSWORD"]
            aws_port = int(st.secrets["MYSQL_PORT"])
        except KeyError:
            aws_host = os.getenv("MYSQL_HOST")
            aws_db = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQL_DB")
            aws_user = os.getenv("MYSQL_USER")
            aws_pass = os.getenv("MYSQL_PASSWORD")
            aws_port = int(os.getenv("MYSQL_PORT", 3306))
        
        # Connect to database
        conn = mysql.connector.connect(
            host=aws_host,
            port=aws_port,
            user=aws_user,
            password=aws_pass,
            database=aws_db,
            connection_timeout=10
        )
        
        cursor = conn.cursor()
        
        # Get timezone information
        cursor.execute("""
            SELECT 
                @@global.time_zone as global_tz,
                @@session.time_zone as session_tz,
                NOW() as db_time,
                UTC_TIMESTAMP() as utc_time,
                CONVERT_TZ(NOW(), @@session.time_zone, '+04:00') as uae_time
        """)
        
        result = cursor.fetchone()
        global_tz, session_tz, db_time, utc_time, uae_time = result
        
        # Get sample data timezone info
        cursor.execute("""
            SELECT 
                MIN(ReceivedAt) as earliest_record,
                MAX(ReceivedAt) as latest_record,
                COUNT(*) as total_records
            FROM sales_data
        """)
        
        data_info = cursor.fetchone()
        earliest, latest, count = data_info
        
        cursor.close()
        conn.close()
        
        # Display timezone analysis
        st.subheader("🕐 Database Timezone Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Database Configuration:**")
            st.write(f"• Global Timezone: `{global_tz}`")
            st.write(f"• Session Timezone: `{session_tz}`")
            st.write(f"• Database Time: `{db_time}`")
            st.write(f"• UTC Time: `{utc_time}`")
            st.write(f"• UAE Time: `{uae_time}`")
        
        with col2:
            st.write("**Data Information:**")
            st.write(f"• Total Records: `{count:,}`")
            st.write(f"• Earliest Record: `{earliest}`")
            st.write(f"• Latest Record: `{latest}`")
        
        # UAE timezone reference
        uae_tz = pytz.timezone('Asia/Dubai')
        uae_now = datetime.now(uae_tz)
        st.write(f"**UAE Reference Time:** `{uae_now.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
        
        # Recommendations
        st.subheader("📋 Recommendations")
        
        if session_tz != '+04:00':
            st.warning("⚠️ **Database timezone is not set to UAE timezone (+04:00)**")
            st.write("**Recommended Actions:**")
            st.write("1. **Database Level Fix (Permanent):** Run the SQL commands in `fix_database_timezone.sql`")
            st.write("2. **Application Level Fix (Current):** The dashboard now forces UAE timezone in connections")
            
            if st.button("🔧 Fix Database Timezone Now"):
                fix_database_timezone()
        else:
            st.success("✅ Database timezone is correctly set to UAE timezone (+04:00)")
        
        return {
            'global_tz': global_tz,
            'session_tz': session_tz,
            'db_time': db_time,
            'utc_time': utc_time,
            'uae_time': uae_time,
            'data_count': count,
            'earliest_record': earliest,
            'latest_record': latest
        }
        
    except Exception as e:
        st.error(f"Error checking database timezone: {e}")
        return None

def fix_database_timezone():
    """
    Fix database timezone to UAE timezone (+04:00)
    """
    try:
        # Get AWS MySQL credentials
        try:
            aws_host = st.secrets["MYSQL_HOST"]
            aws_db = st.secrets["MYSQL_DB"]
            aws_user = st.secrets["MYSQL_USER"]
            aws_pass = st.secrets["MYSQL_PASSWORD"]
            aws_port = int(st.secrets["MYSQL_PORT"])
        except KeyError:
            aws_host = os.getenv("MYSQL_HOST")
            aws_db = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQL_DB")
            aws_user = os.getenv("MYSQL_USER")
            aws_pass = os.getenv("MYSQL_PASSWORD")
            aws_port = int(os.getenv("MYSQL_PORT", 3306))
        
        # Connect to database
        conn = mysql.connector.connect(
            host=aws_host,
            port=aws_port,
            user=aws_user,
            password=aws_pass,
            database=aws_db,
            connection_timeout=10
        )
        
        cursor = conn.cursor()
        
        with st.spinner("🔧 Fixing database timezone..."):
            # Set session timezone to UAE
            cursor.execute("SET time_zone = '+04:00'")
            
            # Try to set global timezone (may require SUPER privilege)
            try:
                cursor.execute("SET GLOBAL time_zone = '+04:00'")
                st.success("✅ Successfully set both session and global timezone to UAE (+04:00)")
            except mysql.connector.Error as e:
                if "Access denied" in str(e):
                    st.warning("⚠️ Cannot set global timezone (requires SUPER privilege). Session timezone set to UAE.")
                    st.info("💡 Contact your database administrator to set global timezone permanently.")
                else:
                    st.error(f"Error setting global timezone: {e}")
            
            # Verify the change
            cursor.execute("SELECT @@session.time_zone, NOW()")
            session_tz, current_time = cursor.fetchone()
            
            st.success(f"✅ Session timezone now set to: {session_tz}")
            st.info(f"🕐 Current database time: {current_time}")
        
        cursor.close()
        conn.close()
        
        # Clear cache to force reload with new timezone
        st.cache_data.clear()
        st.success("🔄 Cache cleared. Data will reload with correct timezone.")
        
    except Exception as e:
        st.error(f"Error fixing database timezone: {e}")

def validate_timezone_consistency():
    """
    Validate that all timestamps are consistent across the system
    """
    st.subheader("🔍 Timezone Consistency Validation")
    
    # Get current times from different sources
    uae_tz = pytz.timezone('Asia/Dubai')
    uae_now = datetime.now(uae_tz)
    
    # Check database time
    db_info = check_database_timezone()
    
    if db_info:
        st.write("**Time Comparison:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("UAE Application Time", uae_now.strftime('%H:%M:%S'))
        
        with col2:
            if db_info['uae_time']:
                db_uae_time = db_info['uae_time'].strftime('%H:%M:%S')
                st.metric("Database UAE Time", db_uae_time)
            else:
                st.metric("Database UAE Time", "N/A")
        
        with col3:
            if db_info['db_time']:
                db_local_time = db_info['db_time'].strftime('%H:%M:%S')
                st.metric("Database Local Time", db_local_time)
            else:
                st.metric("Database Local Time", "N/A")
        
        # Check for consistency
        if db_info['session_tz'] == '+04:00':
            st.success("✅ Timezone consistency validated - all systems using UAE timezone")
        else:
            st.warning("⚠️ Timezone inconsistency detected - application fixes applied but database should be updated")

if __name__ == "__main__":
    st.title("🕐 Timezone Configuration Utility")
    st.write("This utility helps diagnose and fix timezone issues in your sales dashboard.")
    
    tab1, tab2, tab3 = st.tabs(["Check Timezone", "Fix Timezone", "Validate Consistency"])
    
    with tab1:
        if st.button("🔍 Check Database Timezone"):
            check_database_timezone()
    
    with tab2:
        st.write("Use this to fix database timezone configuration:")
        if st.button("🔧 Fix Database Timezone"):
            fix_database_timezone()
    
    with tab3:
        validate_timezone_consistency()
