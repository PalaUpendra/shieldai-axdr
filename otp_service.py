import random, string, smtplib, requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Config — UPDATE THESE BEFORE RUNNING ─────────────
GMAIL_USER     = "your.email@gmail.com"      # ← your Gmail address
GMAIL_PASSWORD = "your-app-password"         # ← Gmail App Password (16 chars)
FAST2SMS_KEY   = "your-fast2sms-api-key"     # ← Fast2SMS API key (for SMS)

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_email_otp(to_email, otp, username):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "ShieldAI — Your Login OTP"
        msg['From']    = GMAIL_USER
        msg['To']      = to_email

        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:0 auto">
          <div style="background:#0f1525;padding:24px;border-radius:12px 12px 0 0">
            <h2 style="color:#63b3ed;margin:0">&#x1F6E1; ShieldAI A-XDR+</h2>
            <p style="color:#94a3b8;margin:4px 0">Autonomous Cyber Defence Platform</p>
          </div>
          <div style="background:#f8fafc;padding:24px;border-radius:0 0 12px 12px;border:1px solid #e2e8f0">
            <p style="color:#2d3748">Hello <b>{username}</b>,</p>
            <p style="color:#4a5568">Your one-time login code is:</p>
            <div style="background:#0f1525;border-radius:8px;padding:20px;text-align:center;margin:20px 0">
              <span style="font-size:36px;font-weight:700;letter-spacing:12px;color:#63b3ed">{otp}</span>
            </div>
            <p style="color:#718096;font-size:13px">
              &#x23F1; This code expires in <b>5 minutes</b><br>
              &#x1F512; Never share this code with anyone<br>
              &#x274C; If you didn't request this, contact your admin
            </p>
          </div>
        </div>
        """
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())
        print(f"Email OTP sent to {to_email} ✅")
        return True
    except Exception as e:
        print(f"Email OTP failed: {e}")
        return False

def send_sms_otp(phone, otp, username):
    try:
        url = "https://www.fast2sms.com/dev/bulkV2"
        payload = {
            "route":            "otp",
            "variables_values": otp,
            "numbers":          phone.replace("+91", "").replace("+", ""),
            "flash":            0
        }
        headers = {
            "authorization": FAST2SMS_KEY,
            "Content-Type":  "application/json"
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        data = r.json()
        if data.get("return"):
            print(f"SMS OTP sent to {phone} ✅")
            return True
        print(f"SMS failed: {data}")
        return False
    except Exception as e:
        print(f"SMS OTP failed: {e}")
        return False

def send_otp(user, method):
    otp = generate_otp()
    success = False
    if method == 'email' and user.email:
        success = send_email_otp(user.email, otp, user.username)
    elif method == 'sms' and user.phone:
        success = send_sms_otp(user.phone, otp, user.username)
    return otp if success else None
