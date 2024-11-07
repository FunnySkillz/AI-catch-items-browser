import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit
import logging

# Set up logging to file
logging.basicConfig(filename="ai_game_log.txt", level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Log file header
logging.info("Starting AI log for game session")

# Set the fixed coordinates for the game area
x, y = 1000, 600
width = 1400
height = 1050
capture_height = height

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold and collection point y-coordinate
drop_threshold_y = y + height - 100
collection_y = 950

# Variables for tracking item positions, speeds, and scores
item_speeds = {}
matching_threshold = 0.5
target_lock_duration = 0.05
pause_frame_threshold = 10
pause_detected = False
score = 0
anticipated_pause_score = [2000, 4000]
pause_start_time = None
pause_in_progress = False

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

def update_item_speeds(current_positions, previous_positions, time_interval):
    """Calculates the speed of each item based on its movement over time."""
    global pause_detected
    speeds = {}
    
    if not pause_detected:
        for pos in current_positions:
            closest_prev = min(previous_positions, key=lambda p: abs(pos[0] - p[0]), default=None)
            if closest_prev and abs(pos[0] - closest_prev[0]) < 10:
                speed = (pos[1] - closest_prev[1]) / time_interval
                speeds[pos[0]] = max(0.01, speed)
    logging.debug(f"Updated item speeds: {speeds}")
    return speeds

def prioritize_item(items):
    """Selects the item closest to the collection point, considering its speed if available."""
    closest_item = None
    min_time_to_reach = float('inf')

    for item in items:
        x, y_pos = item
        distance = abs(collection_y - y_pos)
        speed = item_speeds.get(x, 0.01)
        time_to_reach = distance / speed
        
        if time_to_reach < min_time_to_reach:
            min_time_to_reach = time_to_reach
            closest_item = item

    if closest_item:
        logging.info(f"Targeting item at ({closest_item[0]}, {closest_item[1]}) with estimated time to reach: {min_time_to_reach:.2f}s")
    return closest_item

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the target item using absolute positioning."""
    absolute_x_position = max(x, min(item_x + x, x + width))
    logging.info(f"Attempting to move basket to x-position: {absolute_x_position}")
    pyautogui.moveTo(absolute_x_position, y + height - 30)

def handle_pause():
    """Handles the anticipated pauses at specific scores."""
    global pause_start_time, pause_in_progress
    if pause_start_time is None:
        pause_start_time = time.time()
        pause_in_progress = True
        logging.info("Anticipated pause detected. Waiting for game to resume...")

    if time.time() - pause_start_time >= 3:
        logging.info("Pause over, resuming AI actions.")
        pause_in_progress = False
        return True
    return False

def main():
    global item_speeds, score, pause_start_time, pause_in_progress
    print("Starting AI... Press 'Esc' to stop.")
    logging.info("AI session started.")
    time.sleep(2)

    previous_positions = []
    while True:
        if keyboard.is_pressed("esc"):
            logging.info("AI session ended by user.")
            cv2.destroyAllWindows()
            break

        # Check if we should handle an anticipated pause
        if score in anticipated_pause_score and not pause_in_progress:
            if handle_pause():
                anticipated_pause_score.remove(score)
                pause_start_time = None
            else:
                time.sleep(0.1)
                continue

        screen = capture_screen()
        item_positions = find_items(screen, template, threshold=matching_threshold)

        if not item_positions:
            logging.debug("No items detected.")
            item_speeds = {}
            previous_positions = []
            continue

        # Update item speeds based on current and previous positions
        if previous_positions:
            time_interval = target_lock_duration
            item_speeds = update_item_speeds(item_positions, previous_positions, time_interval)

        # Select the item closest to the collection point, taking speed into account
        target_item = prioritize_item(item_positions)

        # Move only when the target is near the collection point and no pause is detected
        if not pause_detected and target_item and target_item[1] >= collection_y - 50:
            move_to_item_absolute(target_item[0])
            score += 100
            logging.info(f"Score updated: {score}")

        previous_positions = item_positions
        time.sleep(target_lock_duration)

if __name__ == "__main__":
    main()
