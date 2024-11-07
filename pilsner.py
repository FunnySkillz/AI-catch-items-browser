import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit
import logging

# Define the expanded ROI coordinates for better coverage
x, y = 1000, 450
width = 1400
height = 1220
capture_height = height

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold y-coordinate (items below this y cannot be collected)
drop_threshold_y = 950

# Logging setup to save logs on exit
logging.basicConfig(filename="game_ai_log.txt", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Current target tracking
current_target_x = None
target_lock_duration = 0.1  # Time to focus on a target (in seconds)

def capture_screen():
    """Captures a screenshot of the expanded ROI area and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

def find_items(screen, template, threshold=0.6, debug=False):
    """Finds all positions of items in the given screen and returns sorted positions by y-coordinate."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template.shape[1] // 2, loc[1] + template.shape[0] // 2) for loc in zip(*locations[::-1])]

    if debug:
        for loc in zip(*locations[::-1]):
            top_left = (loc[0], loc[1])
            bottom_right = (loc[0] + template.shape[1], loc[1] + template.shape[0])
            cv2.rectangle(screen, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow("Match Debug", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    # Filter items below the drop threshold
    return sorted([pos for pos in positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])

def move_to_item(item_x):
    """Moves the basket horizontally to align with the target item."""
    global current_target_x
    target_position_x = x + item_x
    
    target_position_x = max(x, min(target_position_x, x + width))
    pyautogui.moveTo(target_position_x, y + height - 30)
    current_target_x = item_x
    logging.info(f"Moved basket to {item_x}")

def main():
    global current_target_x
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)
    
    logging.info("AI session started.")
    while True:
        if keyboard.is_pressed("esc"):
            logging.info("Exiting...")
            cv2.destroyAllWindows()
            break

        screen = capture_screen()
        item_positions = find_items(screen, template, threshold=0.6)
        
        if not item_positions:
            logging.debug("No items detected.")
            current_target_x = None
            continue

        # Target the item closest to the drop threshold
        closest_item = min(item_positions, key=lambda item: abs(item[1] - drop_threshold_y))
        
        if current_target_x is None or abs(closest_item[0] - current_target_x) > 10:
            current_target_x = closest_item[0]
            move_to_item(current_target_x)
        
        time.sleep(target_lock_duration)

if __name__ == "__main__":
    main()
