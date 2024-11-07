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
drop_threshold_y = y + height - 100  # Filter out items below this
collection_y = 950  # The y-coordinate where items should be collected

# Initialize variables to track item positions and speeds
item_speeds = {}  # Dictionary to track speeds of items
matching_threshold = 0.5

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
        for prev_pos in previous_positions:
            if abs(pos[0] - prev_pos[0]) < 10:  # Match items by x-coordinate
                speed = (pos[1] - prev_pos[1]) / time_interval
                speeds[pos[0]] = speed  # Store speed using x-coordinate as key
                break
    return speeds

def prioritize_item(items):
    """Selects the item closest to the collection point, considering its speed."""
    closest_item = None
    min_distance = float('inf')
    
    for item in items:
        x, y_pos = item
        if x in item_speeds:  # Check if we have speed data for this item
            time_to_reach = (collection_y - y_pos) / item_speeds[x]
            distance = abs(collection_y - y_pos) / time_to_reach if time_to_reach > 0 else abs(collection_y - y_pos)
        else:
            distance = abs(collection_y - y_pos)
        
        if distance < min_distance:
            min_distance = distance
            closest_item = item
            
    return closest_item

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the target item using absolute positioning."""
    absolute_x_position = x + item_x
    absolute_x_position = max(x, min(absolute_x_position, x + width))
    pyautogui.moveTo(absolute_x_position, y + height - 30)

def main():
    global item_speeds  # Track speeds of items between frames
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

        # Update speeds of detected items
        if previous_positions:
            time_interval = target_lock_duration  # Time between frames
            item_speeds = update_item_speeds(item_positions, previous_positions, time_interval)
        
        # Select the item to prioritize based on closest distance and speed
        target_item = prioritize_item(item_positions)
        
        if target_item and target_item[1] >= collection_y - 50:  # Only move if target is near collection point
            move_to_item_absolute(target_item[0])

        previous_positions = item_positions
        time.sleep(target_lock_duration)

if __name__ == "__main__":
    main()
