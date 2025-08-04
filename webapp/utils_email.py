import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_verification_email(to_email, username, verification_link):
    # Set these as environment variables for security
    GMAIL_USER = os.environ.get('GMAIL_USER')
    GMAIL_PASS = os.environ.get('GMAIL_PASS')
    if not GMAIL_USER or not GMAIL_PASS:
        raise Exception('GMAIL_USER and GMAIL_PASS must be set as environment variables.')

    msg = MIMEMultipart()
    msg['From'] = GMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = 'Verify your email for Breast Cancer Ultrasound App'

    body = f"""
    Hi {username},<br><br>
    Please verify your email by clicking the link below:<br>
    <a href='{verification_link}'>Verify Email</a><br><br>
    If you did not register, please ignore this email.
    """
    msg.attach(MIMEText(body, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, to_email, msg.as_string()) 