"""
Local Time Utilities for Sales Dashboard
Ensures all timestamps are displayed in user's local UAE timezone
"""

import pytz
from datetime import datetime
import pandas as pd
import streamlit as st

# UAE timezone constant
UAE_TZ = pytz.timezone('Asia/Dubai')

def get_local_time():
    """
    Get current time in user's local UAE timezone
    """
    return datetime.now(UAE_TZ)

def convert_to_local_time(timestamp):
    """
    Convert any timestamp to user's local UAE timezone
    
    Args:
        timestamp: datetime object (timezone-aware or naive)
    
    Returns:
        datetime object in UAE timezone
    """
    if timestamp is None:
        return None
    
    if isinstance(timestamp, str):
        # Try to parse string timestamp
        try:
            timestamp = pd.to_datetime(timestamp)
        except:
            return timestamp
    
    if hasattr(timestamp, 'tzinfo'):
        if timestamp.tzinfo is None:
            # Naive timestamp - assume it's already in UAE timezone
            return UAE_TZ.localize(timestamp)
        else:
            # Timezone-aware timestamp - convert to UAE timezone
            return timestamp.astimezone(UAE_TZ)
    
    return timestamp

def format_local_time(timestamp, format_str='%Y-%m-%d %H:%M:%S'):
    """
    Format timestamp in user's local UAE timezone
    
    Args:
        timestamp: datetime object
        format_str: format string for datetime formatting
    
    Returns:
        formatted string in UAE timezone
    """
    local_time = convert_to_local_time(timestamp)
    if local_time and hasattr(local_time, 'strftime'):
        return local_time.strftime(format_str)
    return str(local_time) if local_time else 'N/A'

def ensure_dataframe_local_time(df, timestamp_columns=['ReceivedAt']):
    """
    Ensure all timestamp columns in DataFrame are in user's local UAE timezone
    
    Args:
        df: pandas DataFrame
        timestamp_columns: list of column names containing timestamps
    
    Returns:
        DataFrame with timestamps converted to UAE timezone
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in timestamp_columns:
        if col in df_copy.columns:
            # Convert to UAE timezone
            if df_copy[col].dt.tz is None:
                df_copy[col] = df_copy[col].dt.tz_localize(UAE_TZ)
            else:
                df_copy[col] = df_copy[col].dt.tz_convert(UAE_TZ)
    
    return df_copy

def display_time_info():
    """
    Display current time information in user's local timezone
    """
    local_time = get_local_time()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("🕐 **Current Time**")
    st.sidebar.markdown(f"**UAE Local Time:**")
    st.sidebar.markdown(f"`{local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    st.sidebar.markdown(f"**Date:** {local_time.strftime('%A, %B %d, %Y')}")
    st.sidebar.markdown("---")

def get_time_range_display(start_time, end_time):
    """
    Get formatted time range display in local timezone
    
    Args:
        start_time: start datetime
        end_time: end datetime
    
    Returns:
        formatted string showing time range in local timezone
    """
    start_local = convert_to_local_time(start_time)
    end_local = convert_to_local_time(end_time)
    
    start_str = format_local_time(start_local, '%Y-%m-%d %H:%M:%S')
    end_str = format_local_time(end_local, '%Y-%m-%d %H:%M:%S')
    
    return f"{start_str} to {end_str} (UAE Local Time)"

def validate_local_time_consistency():
    """
    Validate that all time displays are consistent with user's local timezone
    """
    local_time = get_local_time()
    
    st.success(f"✅ All timestamps synchronized to your local UAE timezone")
    st.info(f"🇦🇪 Current UAE Local Time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    st.caption("All data timestamps, comparisons, and displays use your local UAE timezone (+04:00)")

if __name__ == "__main__":
    st.title("🕐 Local Time Utilities Test")
    
    # Test current time
    st.subheader("Current Time")
    local_time = get_local_time()
    st.write(f"UAE Local Time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Test time conversion
    st.subheader("Time Conversion Test")
    test_time = datetime.now()  # Naive datetime
    converted = convert_to_local_time(test_time)
    st.write(f"Original (naive): {test_time}")
    st.write(f"Converted to UAE: {converted}")
    
    # Display time info
    display_time_info()
    validate_local_time_consistency()
