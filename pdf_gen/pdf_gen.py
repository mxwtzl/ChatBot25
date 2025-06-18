from pypdf import PdfReader, PdfWriter
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
import secrets
import string

def generate_user_token(length=10):
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return ''.join(secrets.choice(characters) for _ in range(length))

pdfpath = Path("D:/GitHub/ChatBot25_pdfgen/OnkelBot.pdf")
output_path = Path("D:/GitHub/ChatBot25_pdfgen/OnkelBot_mit_token.pdf")

# === User-ID einfügen ===
user_id = generate_user_token()

# === Original-PDF laden ===
reader = PdfReader(pdfpath)
writer = PdfWriter()

# === Overlay (User-ID als PDF im Speicher) ===
packet = io.BytesIO()
can = canvas.Canvas(packet, pagesize=A4)

# Position anpassen! (x=100, y=100 z.B. links unten)
can.setFont("Courier", 12)
can.drawString(155, 722, f"{user_id}")
can.save()

# Zurück zum Anfang des BytesIO
packet.seek(0)

# === Overlay-PDF lesen ===
overlay_pdf = PdfReader(packet)
overlay_page = overlay_pdf.pages[0]

# === Erste Seite mit Overlay kombinieren ===
original_page = reader.pages[0]
original_page.merge_page(overlay_page)

# === Neue Seite in Writer einfügen ===
writer.add_page(original_page)

# Weitere Seiten hinzufügen, wenn mehrere Seiten vorhanden sind
for i in range(1, len(reader.pages)):
    writer.add_page(reader.pages[i])

# === Ergebnis speichern ===
with open(output_path, "wb") as f_out:
    writer.write(f_out)

print(f"PDF mit User-ID gespeichert unter: {output_path}")