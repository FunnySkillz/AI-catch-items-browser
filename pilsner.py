import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit
import logging

# Configure logging
logging.basicConfig(filename="ai_game_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting AI log for game session")

# Define the expanded ROI based on your observations
x, y = 1000, 450
width = 1400
height = 1220
capture_height = height
basket_width = 100  # Approximate width of the basket in pixels

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold y-coordinate (items below this y cannot be collected)
drop_threshold_y = y + 950
collection_lock_y = y + 900

# Time to focus on a target before considering a new one
target_lock_duration = 0.1

def capture_screen():
    """Captures a screenshot of the expanded ROI area and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

def find_items(screen, template, threshold=0.6):
    """Finds all positions of items in the given screen and returns sorted positions by y-coordinate."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template.shape[1] // 2, loc[1] + template.shape[0] // 2) for loc in zip(*locations[::-1])]
    return sorted([pos for pos in positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])

def move_to_item(item_x, basket_center_x):
    """Moves the basket horizontally only if it's not already aligned with the target."""
    target_x_position = x + item_x - (basket_width // 2)
    if abs(target_x_position - basket_center_x) > 5:  # Move only if misaligned
        pyautogui.moveTo(target_x_position, y + height - 30)
        logging.info(f"Moved basket to x-position: {target_x_position}")
    return target_x_position

def main():
    current_target = None
    basket_center_x = x + width // 2  # Start with basket at center

    logging.info("AI session started.")
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)

    while True:
        if keyboard.is_pressed("esc"):
            logging.info("Exiting...")
            cv2.destroyAllWindows()
            break

        # Capture screen less frequently to optimize performance
        time.sleep(target_lock_duration)
        screen = capture_screen()

        # Find positions of falling items
        item_positions = find_items(screen, template, threshold=0.6)
        
        if item_positions:
            # Select the closest item within the focus zone
            if current_target is None or current_target[1] >= collection_lock_y:
                current_target = item_positions[0]
                logging.info(f"New target acquired at {current_target}")

            # Move basket to align with target only if within collection range
            if current_target[1] >= collection_lock_y:
                basket_center_x = move_to_item(current_target[0], basket_center_x)

            # Reset target if item falls out of collection range
            if current_target[1] >= drop_threshold_y:
                logging.info(f"Target missed: {current_target}")
                current_target = None  # Reset target

        else:
            logging.debug("No items detected.")
            current_target = None  # Reset target if no items are detected

if __name__ == "__main__":
    main()
