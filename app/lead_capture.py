import smtplib
from email.mime.text import MIMEText
from app.config import settings

def send_lead_email(lead_data: dict, message_history: list):
    """
    Sends an email notification with comprehensive lead information.
    
    Args:
        lead_data (dict): A dictionary containing the captured lead's name and email.
        message_history (list): The list of chat messages for this session.
    """
    # Check if all necessary SMTP credentials are set
    if not all([settings.SMTP_SERVER, settings.SMTP_USERNAME, settings.SMTP_PASSWORD, settings.RECEIVER_EMAIL]):
        print("SMTP credentials are not fully configured. Email will not be sent.")
        return

    sender_email = settings.SMTP_USERNAME
    receiver_email = settings.RECEIVER_EMAIL
    
    # Extract the first message as the user's intent
    initial_message = "N/A"
    if message_history:
        initial_message = message_history[0].get("content", "N/A")
    
    # Extract the timestamp of the first message
    timestamp = "N/A"
    if message_history:
        timestamp = message_history[0].get("timestamp", "N/A")
    
    # Create the email message content
    subject = "New OveloAI Lead Captured"
    body = f"""
    A new lead has been captured from your AI assistant!
    
    Name: {lead_data.get('name', 'N/A')}
    Email: {lead_data.get('email', 'N/A')}
    
    ---
    Lead Details:
    Initial Intent: "{initial_message}"
    Time of Capture: {timestamp}
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()  # Upgrade the connection to a secure TLS connection
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Lead email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
