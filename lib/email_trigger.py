def send_email(subject, message):
    sender_email = "ajomonjose123@gmail.com"
    sender_password = "Qwerty@12345"
    receiver_email = ["venkatamanish.canada@gmail.com","ajomonjose123@gmail.com","bhavikg@queenscollege.ca"]

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)

    email_body = f"Subject: {subject}\n\n{message}"
    server.sendmail(sender_email, receiver_email, email_body)

    server.quit()