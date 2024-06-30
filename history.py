import cv2
import pytesseract
import pyautogui
import numpy as np
import time
from PIL import ImageGrab
import pygetwindow as gw
import os

# Update the path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\John\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

WINDOW_TITLE = "Habbo Hotel: Origins"
OUTPUT_FILE = "chat_history.txt"

def capture_window(window_title):
    print("Capturing window...")
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"Window with title '{window_title}' not found.")
    
    window = windows[0]
    print(f"Found window: {window.title} at position {window.left}, {window.top}, {window.right}, {window.bottom}")
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print("Screenshot captured")
    return screenshot

def extract_chat_bubbles(screenshot):
    print("Extracting chat bubbles...")
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours on the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Adjust these parameters based on your needs
        if 50 < w < 400 and 20 < h < 100:
            bubble = screenshot[y:y+h, x:x+w]
            bubbles.append(bubble)
    
    print(f"Found {len(bubbles)} chat bubbles")
    return bubbles

def read_text_from_bubbles(bubbles):
    print("Reading text from bubbles...")
    chat_texts = []
    for bubble in bubbles:
        text = pytesseract.image_to_string(bubble)
        if text.strip():
            chat_texts.append(text.strip())
    print(f"Extracted text from {len(chat_texts)} bubbles")
    return chat_texts

def write_to_file(chat_texts, filename=OUTPUT_FILE):
    try:
        with open(filename, 'a') as file:
            for text in chat_texts:
                file.write(f"{time.ctime()}: {text}\n")
        print(f"Successfully wrote {len(chat_texts)} texts to {filename}")
    except IOError as e:
        print(f"Error writing to file: {e}")

def main():
    last_processed_texts = set()
    
    try:
        while True:
            try:
                print("Start processing cycle...")
                screenshot = capture_window(WINDOW_TITLE)
                
                bubbles = extract_chat_bubbles(screenshot)
                
                chat_texts = read_text_from_bubbles(bubbles)
                
                new_texts = set(chat_texts) - last_processed_texts
                if new_texts:
                    write_to_file(new_texts)
                    last_processed_texts.update(new_texts)
                
                time.sleep(2)
            except Exception as e:
                print(f"Error during processing: {e}")
                time.sleep(5)  # Adjust sleep time to handle retry interval on failure
    except KeyboardInterrupt:
        print("Script terminated by user.")

if __name__ == "__main__":
    # Check if the output file is in a writable directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")
    if os.access(current_directory, os.W_OK):
        print(f"Directory is writable. Starting script...")
        main()
    else:
        print("Output directory is not writable. Check file permissions.")