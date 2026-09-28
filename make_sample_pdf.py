from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

c = canvas.Canvas("data/raw/sample.pdf", pagesize=letter)
c.setFont("Helvetica", 12)
lines = [
    "Security Policy Overview",
    "",
    "All employees must enable two-factor authentication on their",
    "company accounts within 7 days of onboarding. Passwords must",
    "be rotated every 90 days and cannot be reused from the last 5",
    "passwords used.",
    "",
    "Any suspected security incident must be reported to the",
    "security team within 1 hour via the #security-incidents channel.",
]
y = 750
for line in lines:
    c.drawString(50, y, line)
    y -= 20
c.save()
print("Created data/raw/sample.pdf")