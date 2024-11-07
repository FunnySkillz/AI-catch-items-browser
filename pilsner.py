import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit
import logging
from scipy.spatial import KDTree  # Efficient nearest-neighbor search

# Set up logging to file
logging.basicConfig(filename="ai_game_log.txt", level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting AI log for game session")

# Updated ROI coordinates
x, y = 1000, 450
width = 1400
height = 1220
capture_height = height

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Collection thresholds and parameters
collection_y_threshold = 955  # Adjusted for better accuracy
focus_zone_y = 900  # Y-coordinate below which items have priority to minimize switching

# Drop threshold and collection point y-coordinate
drop_threshold_y = y + height - 100

# Variables for tracking game state
matching_threshold = 0.5
target_lock_duration = 0.05
pause_in_progress = False

# Timer-based pause control
last_pause_timestamp = time.time()

def capture_screen():
    """Captures a screenshot of the specified ROI and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    gray_screen = cv2.GaussianBlur(gray_screen, (3, 3), 0)
    return gray_screen

def find_items(screen, template, threshold=matching_threshold):
    """Finds all positions of items in the given screen and returns sorted positions by y-coordinate."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template.shape[1] // 2, loc[1] + template.shape[0] // 2) for loc in zip(*locations[::-1])]
    logging.debug(f"Items detected: {positions}")
    return sorted([pos for pos in positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])

def prioritize_item(items, basket_x):
    """Uses KDTree for efficient nearest-neighbor search to select the closest item."""
    if items:
        # Create a KDTree for quick nearest-neighbor search
        kdtree = KDTree(items)
        
        # Query for the nearest item to the basket x-coordinate
        _, idx = kdtree.query([basket_x, focus_zone_y], k=1)
        closest_item = items[idx]  # Retrieve the closest item position
        
        logging.info(f"Targeting item at ({closest_item[0]}, {closest_item[1]})")
        return closest_item
    return None

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the target item using absolute positioning."""
    absolute_x_position = max(x, min(item_x + x, x + width))
    logging.info(f"Attempting to move basket to x-position: {absolute_x_position}")
    pyautogui.moveTo(absolute_x_position, y + height - 30)

def handle_pause():
    """Handles anticipated pauses based on elapsed time since last pause."""
    global last_pause_timestamp, pause_in_progress
    current_time = time.time()
    
    if current_time - last_pause_timestamp >= 20:  # Assume 20s intervals between known pause points
        pause_in_progress = True
        last_pause_timestamp = current_time
        logging.info("Anticipated pause detected. Waiting for game to resume...")
        time.sleep(3)  # Adjust for observed pause duration
        logging.info("Pause over, resuming AI actions.")
        pause_in_progress = False

def main():
    print("Starting AI... Press 'Esc' to stop.")
    logging.info("AI session started.")
    time.sleep(2)

    basket_x = x + width // 2  # Approximate x-coordinate of the basket's center

    while True:
        if keyboard.is_pressed("esc"):
            logging.info("AI session ended by user.")
            cv2.destroyAllWindows()
            break

        handle_pause()

        screen = capture_screen()
        item_positions = find_items(screen, template, threshold=matching_threshold)

        if not item_positions:
            logging.debug("No items detected.")
            continue

        target_item = prioritize_item(item_positions, basket_x)

        # Move the basket to the target item position if within collection range
        if target_item and target_item[1] >= collection_y_threshold - 5:
            move_to_item_absolute(target_item[0])

        time.sleep(target_lock_duration)

if __name__ == "__main__":
    main()
