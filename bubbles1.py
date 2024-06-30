import os
import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab
import time
import pygetwindow as gw
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\John\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load multiple templates for bubble start and end contours
template_dir = 'grayscale_templates'
start_templates = [cv2.imread(os.path.join(template_dir, f'start_template_{i}.png'), 0) for i in range(1, 3)]
end_templates = [cv2.imread(os.path.join(template_dir, f'end_template_{i}.png'), 0) for i in range(1, 3)]

# Ensure all templates are loaded correctly
if any(template is None for template in start_templates + end_templates):
    raise ValueError("One or more template images could not be loaded. Ensure file paths are correct.")

# Create a folder to save screenshots
screenshot_folder = 'screenshots'
os.makedirs(screenshot_folder, exist_ok=True)

# Function to get the game window bounding box
def get_game_window_bbox():
    game_window = gw.getWindowsWithTitle('Habbo Hotel: Origins')
    if not game_window:
        logging.error("Game window not found. Make sure the window title is correct.")
        return None
    window = game_window[0]
    return (window.left, window.top, window.right, window.bottom)

# Function to capture the game window screen
def capture_game_window():
    bbox = get_game_window_bbox()
    if not bbox:
        raise ValueError("Window bounding box could not be determined.")
    screen = np.array(ImageGrab.grab(bbox=bbox))

    # Save the full screenshot for debugging
    cv2.imwrite(os.path.join(screenshot_folder, 'full_screenshot.png'), cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
    
    # Only focus on the top half of the window for bubble detection
    height = screen.shape[0]
    screen_top_half = screen[0:height//2, :]
    
    return cv2.cvtColor(screen_top_half, cv2.COLOR_RGB2BGR), screen

# Function for template matching with multiple templates
def match_template(image, templates, threshold=0.8):
    points = []
    for template in templates:
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        points.extend(list(zip(*loc[::-1])))
    return points

# Function to detect text bubbles by matching start and end templates
def detect_text_bubbles(screen):
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    # Enhance the grayscale screen for better matching
    gray_screen = cv2.equalizeHist(gray_screen)
    
    # Match using multiple templates
    start_points = match_template(gray_screen, start_templates)
    end_points = match_template(gray_screen, end_templates)
    
    logging.info(f"Start points found: {len(start_points)}")
    logging.info(f"End points found: {len(end_points)}")
    
    bubble_contours = []
    
    for start in start_points:
        for end in end_points:
            if end[0] > start[0] and abs(end[1] - start[1]) < 30:  # Ensure end is after start and on the same y-line
                x, y, w, h = start[0], start[1], end[0] - start[0] + end_templates[0].shape[1], end_templates[0].shape[0]
                bubble_contours.append((x, y, w, h))
                break
    
    return bubble_contours

# Function to preprocess bubble image for OCR
def preprocess_bubble_image(bubble_img):
    gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Function to extract text from detected bubbles
def extract_text_from_bubbles(screen, bubble_contours, detected_bubbles):
    new_texts = []
    for (x, y, w, h) in bubble_contours:
        bubble_img = screen[y:y+h, x:x+w]
        
        # Check if the bubble contour was already detected
        duplicate = False
        for (dx, dy, dw, dh) in detected_bubbles:
            if abs(x - dx) < 10 and abs(y - dy) < 10 and abs(w - dw) < 10 and abs(h - dh) < 10:
                duplicate = True
                break
        
        if not duplicate:
            processed_bubble = preprocess_bubble_image(bubble_img)
            cv2.imwrite(os.path.join(screenshot_folder, f'bubble_debug_{x}_{y}.png'), processed_bubble)  # Save for debugging

            text = pytesseract.image_to_string(processed_bubble, config='--oem 3 --psm 6')
            
            if text.strip():
                new_texts.append(text.strip())
                detected_bubbles.append((x, y, w, h))
                logging.info(f"Extracted text: {text.strip()}")
            else:
                logging.info("No text found in the detected bubble.")
        else:
            logging.info(f"Duplicate bubble not processed: ({x},{y},{w},{h})")
    
    return new_texts

# Function to update detected bubbles in the text file
def update_detected_bubbles(texts):
    if texts:
        with open('history.txt', 'a', encoding='utf-8') as file:
            for i, text in enumerate(texts):
                file.write(f"Bubble {i+1}: {text}\n")
                file.write("-" * 20 + "\n")
    else:
        logging.info("No new bubbles detected or text extracted.")

# Main function to capture and process chat bubbles
def capture_and_process_chat():
    detected_bubbles = []
    while True:
        try:
            # Capture both the top half of the screen and the full screenshot
            screen_top_half, full_screen = capture_game_window()
            
            # Save the full grayscale screenshot for debugging
            cv2.imwrite(os.path.join(screenshot_folder, 'full_screenshot_gray.png'), cv2.cvtColor(full_screen, cv2.COLOR_RGB2GRAY))

            # Detect bubbles in the top half of the screen
            bubble_contours = detect_text_bubbles(screen_top_half)

            # Draw detected contours on the full screenshot for visualization
            for (x, y, w, h) in bubble_contours:
                x_full, y_full = x, y  # Since we're only looking at the top half, x, y coordinates are the same
                cv2.rectangle(full_screen, (x_full, y_full), (x_full + w, y_full + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(screenshot_folder, 'full_screenshot_with_contours.png'), cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR))

            if bubble_contours:
                new_texts = extract_text_from_bubbles(screen_top_half, bubble_contours, detected_bubbles)
                
                if new_texts:
                    update_detected_bubbles(new_texts)
                else:
                    logging.info("No new text found in bubbles detected.")
            else:
                logging.info("No bubbles detected.")
                
            time.sleep(1)  # Take a screenshot every second
        
        except ValueError as e:
            logging.error(f"Error: {e}")
            break
        except Exception as e:
            logging.exception("Unexpected error occurred.")
            break

# Run the main function
if __name__ == "__main__":
    capture_and_process_chat()