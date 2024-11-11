import cv2
import pyautogui
import numpy as np
import time
import keyboard
import logging
from scipy.spatial import distance

# Configure detailed logging
logging.basicConfig(filename="game_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting AI log with direct movement implementation")

# Define the Region of Interest (ROI)
x, y = 1000, 450
width = 1400
height = 1220
capture_height = height

# Load and prepare the template image for the beer bottle in grayscale
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold y-coordinate
drop_threshold_y = y + 955

# Minimum distance between items to consider them unique
MIN_DISTANCE = 50

# Lock threshold
LOCK_Y_THRESHOLD = y + 850

# Initialize a dictionary to keep track of items with unique IDs
item_tracker = {}
next_item_id = 1
locked_item_id = None
adaptive_capture_interval = 0.05

def capture_screen():
    """Captures a screenshot of the expanded ROI area and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

def find_items(screen, template, threshold=0.55):
    """Finds all positions of items in the given screen and returns filtered unique positions."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template_width // 2, loc[1] + template_height // 2) for loc in zip(*locations[::-1])]
    
    unique_positions = []
    for pos in positions:
        if all(distance.euclidean(pos, uniq_pos) > MIN_DISTANCE for uniq_pos in unique_positions):
            unique_positions.append(pos)
    
    sorted_positions = sorted([pos for pos in unique_positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])
    logging.debug(f"Unique detected item positions: {sorted_positions}")
    return sorted_positions

def track_items(detected_positions):
    """Track detected items and assign or update unique IDs."""
    global next_item_id
    updated_tracker = {}

    for pos in detected_positions:
        found_match = False
        for item_id, item_pos in item_tracker.items():
            if distance.euclidean(pos, item_pos) < MIN_DISTANCE:
                updated_tracker[item_id] = pos
                found_match = True
                break

        if not found_match:
            updated_tracker[next_item_id] = pos
            logging.info(f"New item detected with ID: {next_item_id} at {pos}")
            next_item_id += 1

    # Log the updated tracker information for each frame
    logging.info(f"Updated Tracker: {updated_tracker}")
    return updated_tracker

def select_closest_item(tracked_items):
    """Selects the item closest to the basket to prioritize, maintaining lock if within threshold."""
    global locked_item_id
    if not tracked_items:
        return None

    # If we have a locked item and it’s still within range, keep it locked
    if locked_item_id in tracked_items and tracked_items[locked_item_id][1] < drop_threshold_y:
        logging.info(f"Maintaining lock on item ID {locked_item_id} at position {tracked_items[locked_item_id]}")
        return locked_item_id, tracked_items[locked_item_id]

    # Otherwise, select a new closest item by y-coordinate
    closest_item = min(tracked_items.items(), key=lambda item: item[1][1])

    # Lock onto this item if it’s close enough
    if closest_item[1][1] >= LOCK_Y_THRESHOLD:
        locked_item_id = closest_item[0]
        logging.info(f"Locking onto new item ID {locked_item_id} at position {closest_item[1]}")
    else:
        locked_item_id = None

    return closest_item

def move_basket_direct(item_x):
    """Moves the basket directly to the x-coordinate of the item."""
    basket_x = x + item_x
    pyautogui.moveTo(basket_x, y + height - 30)
    logging.info(f"Moved basket directly to x-position: {basket_x}")

def main():
    logging.info("AI session started with direct movement approach for the basket.")
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
        item_positions = find_items(screen, template, threshold=0.55)
        
        # Track and update item positions
        global item_tracker
        item_tracker = track_items(item_positions)

        # Select the closest item based on priority
        closest_item = select_closest_item(item_tracker)
        
        if closest_item:
            item_id, (item_x, item_y) = closest_item
            logging.info(f"Targeting item ID {item_id} at position {item_x}, {item_y}")
            
            # Move the basket directly to align with the closest item
            move_basket_direct(item_x)

            # Adaptive capture interval adjustment
            global adaptive_capture_interval
            adaptive_capture_interval = 0.03 if item_y >= LOCK_Y_THRESHOLD else 0.05
        else:
            logging.info("No items detected or all items out of range.")

        # Pause based on adaptive capture frequency
        time.sleep(adaptive_capture_interval)

if __name__ == "__main__":
    main()
