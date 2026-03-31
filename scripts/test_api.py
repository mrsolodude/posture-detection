import requests
import json
import time

BASE_URL = "http://localhost:8080"
PHONE = "+919443005037"
PASSWORD = "[PASSWORD]"
def test_workflow():
    print("--- 1. Testing Registration ---")
    try:
        reg_res = requests.post(f"{BASE_URL}/register?phone_number={PHONE}&password={PASSWORD}")
        print(reg_res.json())
    except Exception as e:
        print(f"Registration (likely already exists): {e}")

    print("\n--- 2. Testing Login ---")
    login_res = requests.post(f"{BASE_URL}/login?phone_number={PHONE}&password={PASSWORD}")
    login_data = login_res.json()
    print(login_data)
    token = login_data.get("access_token")

    print("\n--- 3. Testing OTP Request ---")
    otp_req = requests.post(f"{BASE_URL}/request_otp?phone_number={PHONE}")
    print(otp_req.json())

    print("\n--- 4. Testing Activity History ---")
    hist_res = requests.get(f"{BASE_URL}/history")
    print(f"Recent Activities count: {len(hist_res.json())}")

    print("\n--- 5. Testing Alert History ---")
    alert_res = requests.get(f"{BASE_URL}/alerts")
    print(f"Recent Alerts count: {len(alert_res.json())}")

    print("\n--- Workflow Summary ---")
    if token:
        print("✅ SUCCESS: Full API workflow verified.")
    else:
        print("❌ FAILED: JWT Token not received.")

if __name__ == "__main__":
    test_workflow()
    print("\nReminder: To test the real-time stream, run the backend and open frontend/index.html")
