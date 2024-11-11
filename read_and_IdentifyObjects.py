import cv2
import pyautogui
import numpy as np
import time
import keyboard
import logging
from scipy.spatial import distance

# Configure logging
logging.basicConfig(filename="game_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting AI log with object tracking, prioritization, and target lock")

# Define the Region of Interest (ROI) based on your observations
x, y = 1000, 450
width = 1400
height = 1220
capture_height = height

# Load and prepare the template image for the beer bottle in grayscale
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold y-coordinate (items below this y are considered caught)
drop_threshold_y = y + 955  # Adjusted for better catching

# Minimum distance between items to consider them unique
MIN_DISTANCE = 50

# Lock threshold to keep the AI focused on a target item once it's close enough to the basket
LOCK_Y_THRESHOLD = y + 850  # Adjust this based on testing to prevent premature switching

# Initialize a dictionary to keep track of items with unique IDs
item_tracker = {}
next_item_id = 1  # Incremental ID for each new detected item
locked_item_id = None  # Holds the ID of the currently locked item

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

    return updated_tracker

def select_closest_item(tracked_items):
    """Selects the item closest to the basket to prioritize."""
    global locked_item_id
    if not tracked_items:
        return None

    # If we have a locked item and it’s still within range, keep it locked
    if locked_item_id in tracked_items and tracked_items[locked_item_id][1] < drop_threshold_y:
        return locked_item_id, tracked_items[locked_item_id]

    # Otherwise, select a new closest item by y-coordinate
    closest_item = min(tracked_items.items(), key=lambda item: item[1][1])  # Min by y-coordinate

    # Lock onto this item if it’s close enough
    if closest_item[1][1] >= LOCK_Y_THRESHOLD:
        locked_item_id = closest_item[0]
    else:
        locked_item_id = None  # Reset lock if no item is close enough

    return closest_item

def move_basket_to_item(item_x):
    """Moves the basket to the x-coordinate of the item."""
    basket_x = x + item_x
    pyautogui.moveTo(basket_x, y + height - 30)
    logging.info(f"Moved basket to x-position: {basket_x}")

def main():
    logging.info("AI session started for detecting items, tracking, and target lock.")
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
        
        # Track and update item positions
        global item_tracker
        item_tracker = track_items(item_positions)
        logging.info(f"Tracked items: {item_tracker}")

        # Select the closest item based on priority (smallest y-coordinate)
        closest_item = select_closest_item(item_tracker)
        
        if closest_item:
            item_id, (item_x, item_y) = closest_item
            logging.info(f"Targeting item ID {item_id} at position {item_x}, {item_y}")

            # Move basket to align with closest item
            move_basket_to_item(item_x)
        else:
            logging.info("No items detected or all items out of range.")

        # Pause briefly to avoid high CPU usage
        time.sleep(0.03)

if __name__ == "__main__":
    main()
