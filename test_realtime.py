"""
Test Real-Time Local Time Display
Quick test to verify current time is working correctly
"""

import streamlit as st
import pytz
from datetime import datetime
import time

# UAE timezone
UAE_TZ = pytz.timezone('Asia/Dubai')

st.title("🕐 Real-Time Local Time Test")

st.write("This test verifies that the dashboard shows your current real-time local time.")

# Current time display
current_time = datetime.now(UAE_TZ)
st.subheader(f"Current UAE Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Expected time based on system metadata
st.write("**Expected Time (from system):** 2025-07-15T08:56:22+04:00")
st.write(f"**Actual Time (from code):** {current_time.isoformat()}")

# Time comparison
expected_hour = 8
expected_minute = 56
actual_hour = current_time.hour
actual_minute = current_time.minute

st.write("---")
st.subheader("Time Validation:")

if actual_hour == expected_hour:
    st.success(f"✅ Hour matches: {actual_hour}")
else:
    st.error(f"❌ Hour mismatch: Expected {expected_hour}, Got {actual_hour}")

if abs(actual_minute - expected_minute) <= 2:  # Allow 2-minute tolerance
    st.success(f"✅ Minute close enough: {actual_minute} (expected ~{expected_minute})")
else:
    st.error(f"❌ Minute mismatch: Expected ~{expected_minute}, Got {actual_minute}")

# Auto-refresh every 5 seconds
st.write("---")
st.write("**Auto-refresh test:** This page should update automatically")

# Add refresh button
if st.button("🔄 Refresh Now"):
    st.rerun()

# Show multiple time formats
st.write("---")
st.subheader("Time Formats:")
st.write(f"**ISO Format:** {current_time.isoformat()}")
st.write(f"**Display Format:** {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.write(f"**12-Hour Format:** {current_time.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
st.write(f"**Date Only:** {current_time.strftime('%A, %B %d, %Y')}")

# Timezone info
st.write("---")
st.subheader("Timezone Information:")
st.write(f"**Timezone:** {current_time.tzinfo}")
st.write(f"**UTC Offset:** {current_time.strftime('%z')}")
st.write(f"**Timezone Name:** {current_time.tzname()}")

# JavaScript-based real-time clock (if needed)
st.write("---")
st.subheader("JavaScript Real-Time Clock:")
st.components.v1.html("""
<div id="clock" style="font-size: 24px; font-weight: bold; color: #1f77b4;"></div>
<script>
function updateClock() {
    const now = new Date();
    // Convert to UAE timezone (UTC+4)
    const uaeTime = new Date(now.getTime() + (4 * 60 * 60 * 1000));
    const timeString = uaeTime.toISOString().slice(0, 19).replace('T', ' ') + ' +04:00';
    document.getElementById('clock').innerHTML = 'JS Clock: ' + timeString;
}
updateClock();
setInterval(updateClock, 1000);
</script>
""", height=100)

st.write("**Note:** If the Python time and JavaScript time don't match, there might be a system timezone issue.")
