import cv2
import os

# Directory containing original templates
input_dir = 'templates'
# Directory to save grayscale templates
output_dir = 'grayscale_templates'
os.makedirs(output_dir, exist_ok=True)

# List of template filenames
templates = ['start_template_1.png', 'start_template_2.png', 'end_template_1.png', 'end_template_2.png']

for template in templates:
    # Load the colored template image
    img = cv2.imread(os.path.join(input_dir, template))
    if img is None:
        print(f"Error loading image: {template}")
        continue
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale image
    cv2.imwrite(os.path.join(output_dir, template), gray)
    print(f'Saved grayscale image: {template}')