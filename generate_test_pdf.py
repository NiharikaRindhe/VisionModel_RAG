import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import os

def create_chart_image(filename):
    # Create a simple bar chart using PIL
    width = 400
    height = 300
    background_color = "white"
    
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Draw axes
    draw.line([(50, 250), (350, 250)], fill="black", width=2) # X
    draw.line([(50, 250), (50, 50)], fill="black", width=2)   # Y
    
    # Data: Q1: 200, Q2: 450, Q3: 300, Q4: 600
    data = [200, 450, 300, 600]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    colors = ["red", "blue", "green", "orange"]
    
    bar_width = 40
    spacing = 30
    start_x = 70
    
    max_val = 800
    scale = 200 / max_val # 200 pixels height for max_val
    
    for i, val in enumerate(data):
        h = val * (200 / 800)
        x = start_x + (bar_width + spacing) * i
        y = 250 - h
        
        draw.rectangle([x, y, x + bar_width, 250], fill=colors[i])
        
        # Label
        # draw.text((x + 10, 255), quarters[i], fill="black") # default font
        # Trying to use default font
        draw.text((x + 5, 255), quarters[i], fill="black")
        draw.text((x + 5, y - 15), str(val), fill="black")
        
    draw.text((150, 20), "Quarterly Revenue 2024", fill="black")
    
    img.save(filename)
    print(f"Created image: {filename}")

def create_pdf(pdf_name, image_name):
    doc = fitz.open()
    page = doc.new_page()
    
    # Add Title
    page.insert_text((50, 50), "Annual Financial Report 2024", fontsize=18)
    
    # Add Context
    text = (
        "This document contains the financial summary for the fiscal year 2024. "
        "The company experienced significant growth in the fourth quarter. "
        "The visual data below provides a breakdown of revenue by quarter."
    )
    page.insert_textbox((50, 80, 550, 150), text, fontsize=12)
    
    # Insert Image
    rect = fitz.Rect(50, 160, 450, 460) # x0, y0, x1, y1
    page.insert_image(rect, filename=image_name)
    
    # Add analysis text
    analysis = (
        "Analysis:\n"
        "1. Q4 was the strongest quarter with 600 units.\n"
        "2. Q1 was the weakest with 200 units.\n"
        "3. Growth was observed from Q3 to Q4."
    )
    page.insert_textbox((50, 480, 550, 600), analysis, fontsize=12)
    
    output_path = os.path.join("documents", pdf_name)
    if not os.path.exists("documents"):
        os.makedirs("documents")
        
    doc.save(output_path)
    print(f"Created PDF: {output_path}")

try:
    img_filename = "temp_chart_ag.png"
    create_chart_image(img_filename)
    create_pdf("financial_report_2024.pdf", img_filename)
    
    # Cleanup image
    if os.path.exists(img_filename):
        os.remove(img_filename)
        
except ImportError as e:
    print(f"Missing dependency: {e}")
except Exception as e:
    print(f"Error: {e}")
