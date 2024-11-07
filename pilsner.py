import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit

# Set the fixed coordinates for the game area
x, y = 1000, 600
width = 1400
height = 1050
capture_height = height

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold and collection point y-coordinate
drop_threshold_y = y + height - 100  # Filter out items below this threshold
collection_y = 950  # The y-coordinate where items should be collected

# Variables for tracking item positions and speeds
item_speeds = {}  # Dictionary to track speeds of items
matching_threshold = 0.5
target_lock_duration = 0.1  # Time to stay focused on a target (in seconds)

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
    return sorted([pos for pos in positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])

def update_item_speeds(current_positions, previous_positions, time_interval):
    """Calculates the speed of each item based on its movement over time."""
    speeds = {}
    for pos in current_positions:
        # Find the closest previous position by x-coordinate to match the item
        closest_prev = min(previous_positions, key=lambda p: abs(pos[0] - p[0]), default=None)
        if closest_prev and abs(pos[0] - closest_prev[0]) < 10:  # Match items by proximity in x
            speed = (pos[1] - closest_prev[1]) / time_interval
            speeds[pos[0]] = max(0, speed)  # Store speed; ensure non-negative
    return speeds

def prioritize_item(items):
    """Selects the item closest to the collection point, considering its speed if available."""
    closest_item = None
    min_time_to_reach = float('inf')

    for item in items:
        x, y_pos = item
        distance = abs(collection_y - y_pos)
        time_to_reach = distance / item_speeds.get(x, 1)  # Use speed if available, else assume speed of 1
        
        if time_to_reach < min_time_to_reach:
            min_time_to_reach = time_to_reach
            closest_item = item

    return closest_item

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the target item using absolute positioning."""
    absolute_x_position = max(x, min(x + item_x, x + width))
    pyautogui.moveTo(absolute_x_position, y + height - 30)

def main():
    global item_speeds
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)

    previous_positions = []
    while True:
        if keyboard.is_pressed("esc"):
            print("Exiting...")
            cv2.destroyAllWindows()
            break
        
        screen = capture_screen()
        item_positions = find_items(screen, template, threshold=matching_threshold)

        if not item_positions:
            item_speeds = {}
            previous_positions = []
            continue

        # Update item speeds based on current and previous positions
        if previous_positions:
            time_interval = target_lock_duration  # Time between frames
            item_speeds = update_item_speeds(item_positions, previous_positions, time_interval)

        # Select the item closest to the collection point, taking speed into account
        target_item = prioritize_item(item_positions)

        # Move only when the target is near the collection point
        if target_item and target_item[1] >= collection_y - 50:
            move_to_item_absolute(target_item[0])

        previous_positions = item_positions
        time.sleep(target_lock_duration)

if __name__ == "__main__":
    main()
