import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit
import logging
from scipy.spatial import distance

# Configure logging
logging.basicConfig(filename="game_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting AI log for continuous item detection")

# Define the Region of Interest (ROI) based on your observations
x, y = 1000, 450
width = 1400
height = 1220
capture_height = height

# Load and prepare the template image for the beer bottle in grayscale
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold y-coordinate (items below this y cannot be collected)
drop_threshold_y = y + 950

# Minimum distance between items to consider them unique
MIN_DISTANCE = 50

def capture_screen():
    """Captures a screenshot of the expanded ROI area and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

def find_items(screen, template, threshold=0.6):
    """Finds all positions of items in the given screen and returns filtered unique positions."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template_width // 2, loc[1] + template_height // 2) for loc in zip(*locations[::-1])]
    
    # Filter for unique positions based on minimum distance
    unique_positions = []
    for pos in positions:
        if all(distance.euclidean(pos, uniq_pos) > MIN_DISTANCE for uniq_pos in unique_positions):
            unique_positions.append(pos)
    
    sorted_positions = sorted([pos for pos in unique_positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])
    logging.debug(f"Unique detected item positions: {sorted_positions}")
    return sorted_positions

def move_basket_to_item(item_x):
    """Moves the basket to the x-coordinate of the item."""
    basket_x = x + item_x  # Convert item_x relative to screen's absolute x-coordinate
    pyautogui.moveTo(basket_x, y + height - 30)  # Align with the item horizontally near the bottom of the game area
    logging.info(f"Moved basket to x-position: {basket_x}")

def main():
    logging.info("AI session started for detecting items and basic basket movement.")
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)

    while True:
        if keyboard.is_pressed("esc"):
            logging.info("Exiting...")
            cv2.destroyAllWindows()
            break

        # Capture screen to locate items
        screen = capture_screen()

        # Find unique positions of falling items
        item_positions = find_items(screen, template, threshold=0.6)
        
        # Log all unique detected items
        if item_positions:
            logging.info(f"Detected unique items: {item_positions}")
            closest_item = item_positions[0]  # Select the closest item (smallest y-coordinate)
            
            # Move basket to align with closest item
            move_basket_to_item(closest_item[0])

        else:
            logging.info("No items detected.")

        # Pause briefly to avoid high CPU usage
        time.sleep(0.03)

if __name__ == "__main__":
    main()
