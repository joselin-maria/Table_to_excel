import os  
from pdf2image import convert_from_path  
  
# Set the PDF file path  
pdf_path = 'test.pdf'  
  
# Convert the first page of the PDF to a JPEG image  
first = 14
last = 14
images = convert_from_path(pdf_path, dpi=300, first_page=first, last_page=last, poppler_path=r"C:\poppler-23.07.0\Library\bin")  
  
# Save the image file  
image_path = os.path.splitext(pdf_path)[0]

for index, image in enumerate(images):
    image.save(image_path + "p" + str(index+first) + '.jpg', 'JPEG')  