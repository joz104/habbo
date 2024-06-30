import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab
import time
import pygetwindow as gw  # Ensure you have installed pygetwindow (`pip install pygetwindow`)

# Set the path to Tesseract executable (you need to have Tesseract installed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\John\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Function to get the game window bounding box
def get_game_window_bbox():
    # Replace 'Habbo Hotel: Origins' with your specific window title
    game_window = gw.getWindowsWithTitle('Habbo Hotel: Origins')[0]
    left, top, right, bottom = game_window.left, game_window.top, game_window.right, game_window.bottom
    return (left, top, right, bottom)

# Function to capture the game window screen
def capture_game_window():
    bbox = get_game_window_bbox()
    screen = np.array(ImageGrab.grab(bbox=bbox))
    return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

# Function to detect white bubbles in the screen
def detect_white_bubbles(screen):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if w > 50 and h > 20 and aspect_ratio > 2:
            bubble_contours.append((x, y, w, h))
    
    return bubble_contours

# Function to extract text from detected bubbles
def extract_text_from_bubbles(screen, bubble_contours):
    texts = []
    for (x, y, w, h) in bubble_contours:
        bubble_img = screen[y:y+h, x:x+w+4500]
        text = pytesseract.image_to_string(bubble_img)
        if text.strip():
            texts.append(text.strip())
    return texts

# Function to update detected bubbles
def update_detected_bubbles(detected_bubbles):
    with open('history.txt', 'a', encoding='utf-8') as file:
        for i, (x, y, w, h, text) in enumerate(detected_bubbles):
            file.write(f"Bubble {i+1}: {text}\n")
            file.write("-" * 20 + "\n")

# Main function to capture and process chat bubbles
def capture_and_process_chat():
    while True:
        try:
            screen = capture_game_window()
            bubble_contours = detect_white_bubbles(screen)
            detected_bubbles = [(x, y, w, h, extract_text_from_bubbles(screen, [(x, y, w, h)])) for (x, y, w, h) in bubble_contours]
            update_detected_bubbles(detected_bubbles)
        except IndexError:
            print("Window not found. Make sure the window title is correct.")
            break
        time.sleep(1)  # Adjust the delay as needed

# Run the main function
if __name__ == "__main__":
    capture_and_process_chat()
