import os
import re
import pandas as pd
import msoffcrypto
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from playwright.sync_api import sync_playwright

# === CONFIG ===
BASE_DIR = r"C:\Users\sreer\OneDrive\Desktop\mysql"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
load_dotenv(os.path.join(BASE_DIR, ".env"))

USERNAME = os.getenv("GRUBTECH_USERNAME")
PASSWORD = os.getenv("GRUBTECH_PASSWORD")
EXCEL_PASSWORD = os.getenv("GRUBTECH_EXCEL_PASSWORD")

engine = create_engine(
    "mysql+pymysql://root:Nbysccvp%4099@127.0.0.1:3306/order_level",
    connect_args={"charset": "utf8mb4"}
)

# === UTILITIES ===
def normalize_column(col):
    col = col.strip()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"__+", "_", col)
    return col.strip("_").lower()

column_map = {
    "brand": "Brand",
    "channel": "Channel",
    "location": "Location",
    "unique_order_id": "UniqueOrderID",
    "order_id": "OrderID",
    "sequence_number": "SequenceNumber",
    "received_at": "ReceivedAt",
    "type": "Type",
    "customer_name": "CustomerName",
    "telephone": "Telephone",
    "address": "Address",
    "vat_id": "VATID",
    "currency": "Currency",
    "item_price": "ItemPrice",
    "surcharge": "Surcharge",
    "delivery": "Delivery",
    "net_sales": "NetSales",
    "gross_price": "GrossPrice",
    "discount": "Discount",
    "vat": "VAT",
    "total_receipt_total": "Total",
    "channel_service_charge": "ChannelServiceCharge",
    "payment_method": "PaymentMethod",
    "payment_type": "PaymentType",
    "fort_id": "FortID",
    "discount_code": "DiscountCode",
    "delivery_partner_name": "DeliveryPartnerName",
    "delivery_plan": "DeliveryPlan",
    "note": "Note",
    "customer_note": "CustomerNote",
    "employee_name": "EmployeeName",
    "tips": "Tips",
    "sourcesheet": "SourceSheet"
}

# === Step 1: Download Encrypted File ===
def fetch_orders_excel():
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        page = None
        try:
            print("🔐 Logging into Grubtech...")
            browser = p.chromium.launch(headless=True, slow_mo=50)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            page.goto("https://grubcenter.grubtech.io/", timeout=60000)
            page.fill('#email', USERNAME)
            page.fill('#password', PASSWORD)
            page.click('#login')
            page.wait_for_load_state('networkidle', timeout=60000)

            page.goto("https://grubcenter.grubtech.io/realtime-reports/sales/orders", timeout=60000)
            page.wait_for_selector("div.reportPage__filterButton", timeout=60000)
            page.click("div.reportPage__filterButton")
            page.fill('input[placeholder="Select Country"]', "United Arab Emirates")
            page.click("li:has-text('United Arab Emirates')")
            page.evaluate("document.querySelector('#reporting-filter-panel #apply-button').click()")
            page.wait_for_load_state('networkidle', timeout=90000)

            page.locator("span.datePicker__dateText").click()
            page.locator("p:has-text('Today')").click(force=True)
            page.locator("#gt-date-picker-container button:has-text('Apply')").click(force=True)
            page.wait_for_timeout(12000)

            page.wait_for_function("""
                () => [...document.querySelectorAll('button')].some(b => b.innerText.trim().toLowerCase() === 'download' && !b.disabled)
            """, timeout=30000)
            page.click("button:has-text('Download')")

            page.locator('svg[data-testid="CheckCircleSharpIcon"]').wait_for(state='visible', timeout=120000)
            with page.expect_download(timeout=120000) as download_info:
                page.locator('span.downloadPopup__element__statusIcon').click(force=True)
            download = download_info.value

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(DOWNLOAD_DIR, f"grubtech_orders_{timestamp}.xlsx")
            download.save_as(final_path)
            print(f"📥 Download complete: {final_path}")
            return final_path

        except Exception as e:
            print(f"❌ Download Error: {e}")
            if page:
                try:
                    page.screenshot(path=os.path.join(DOWNLOAD_DIR, "error_screenshot.png"))
                except: pass
            return None

# === Step 2: Decrypt + Load Excel ===
def load_protected_excel(filepath):
    decrypted = BytesIO()
    with open(filepath, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=EXCEL_PASSWORD)
        office_file.decrypt(decrypted)
    df = pd.read_excel(decrypted, engine="openpyxl")

    df.columns = [normalize_column(col) for col in df.columns]
    df.rename(columns=column_map, inplace=True)

    df["OrderID"] = df["OrderID"].astype(str)
    df["SequenceNumber"] = df["SequenceNumber"].astype(str)
    df["UniqueOrderID"] = df["UniqueOrderID"].astype("Int64")
    df["ReceivedAt"] = pd.to_datetime(df["ReceivedAt"], errors="coerce")
    df.dropna(subset=["ReceivedAt"], inplace=True)
    return df

# === Step 3: Upload to MySQL ===
def upload_to_mysql(df):
    if "ReceivedAt" not in df.columns or df["ReceivedAt"].isna().all():
        print("⚠️ Skipping delete step: no valid 'ReceivedAt' timestamps found in the upload.")
    else:
        df["ReceivedDate"] = df["ReceivedAt"].dt.date
        unique_dates = df["ReceivedDate"].unique()

        with engine.begin() as conn:
            for d in unique_dates:
                result = conn.execute(
                    text(f"DELETE FROM order_level.sales_data WHERE DATE(ReceivedAt) = '{d}'")
                )
                if result.rowcount == 0:
                    print(f"🧹 No existing records found for {d}. Skipping delete.")
                else:
                    print(f"🧼 Deleted {result.rowcount} existing record(s) for {d}.")

        df.drop(columns=["ReceivedDate"], inplace=True)

    df.to_sql(
        name="sales_data",
        con=engine,
        schema="order_level",
        if_exists="append",
        index=False
    )
    print(f"✅ Uploaded {len(df)} row(s) to MySQL.")

# === Run It All ===
if __name__ == "__main__":
    print("🚀 Starting full pipeline...")
    file_path = fetch_orders_excel()
    if file_path:
        try:
            df = load_protected_excel(file_path)
            upload_to_mysql(df)
            print("🎉 All done! Today’s Grubtech data is safely in MySQL.")
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted local file: {file_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete 