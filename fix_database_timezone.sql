-- Fix Database Timezone Configuration for UAE (Asia/Dubai)
-- Run these commands on your MySQL database to set the timezone to UAE

-- 1. Check current timezone settings
SELECT @@global.time_zone, @@session.time_zone, NOW() as current_time;

-- 2. Set global timezone to UAE (Asia/Dubai = +04:00)
SET GLOBAL time_zone = '+04:00';

-- 3. Set session timezone to UAE
SET time_zone = '+04:00';

-- 4. Verify the change
SELECT @@global.time_zone, @@session.time_zone, NOW() as current_time_uae;

-- 5. Optional: Make this permanent by adding to MySQL configuration file (my.cnf or my.ini)
-- Add this line under [mysqld] section:
-- default-time-zone = '+04:00'

-- 6. Test with sample data to verify timezone is working correctly
SELECT 
    ReceivedAt,
    CONVERT_TZ(ReceivedAt, '+00:00', '+04:00') as ReceivedAt_UAE,
    DATE(ReceivedAt) as Date_Local,
    TIME(ReceivedAt) as Time_Local
FROM sales_data 
ORDER BY ReceivedAt DESC 
LIMIT 5;
