import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit

# Set the fixed coordinates for the game area based on your latest values
x, y = 1000, 600
width = 1400  # Covers the full horizontal game area
height = 1050  # Covers the full vertical game area
capture_height = height  # Set capture height to cover the full vertical game area

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Drop threshold y-coordinate (items below this y cannot be collected)
drop_threshold_y = y + height - 100  # Set based on your observations; adjust if needed

# Initialize variables to track the currently targeted item
current_target_x = None
target_lock_duration = 0.1  # Time to stay focused on a target (in seconds)

# Lower threshold for better sensitivity in matching
matching_threshold = 0.5  # Lowered threshold for improved detection

def capture_screen():
    """Captures a screenshot of the specified ROI and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # Apply slight Gaussian blur to reduce noise and improve matching consistency
    gray_screen = cv2.GaussianBlur(gray_screen, (3, 3), 0)
    return gray_screen

def find_items(screen, template, threshold=matching_threshold):
    """Finds all positions of items in the given screen and returns sorted positions by y-coordinate."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template.shape[1] // 2, loc[1] + template.shape[0] // 2) for loc in zip(*locations[::-1])]
    # Filter out items below the drop threshold
    return sorted([pos for pos in positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the target item using absolute positioning."""
    # Calculate the absolute x-position within the game area
    absolute_x_position = x + item_x
    
    # Limit the x-position to stay within the game area's width
    absolute_x_position = max(x, min(absolute_x_position, x + width))

    pyautogui.moveTo(absolute_x_position, y + height - 30)  # Move to item_x in game area near bottom

def main():
    global current_target_x  # Track the current target
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)

    while True:
        if keyboard.is_pressed("esc"):
            print("Exiting...")
            cv2.destroyAllWindows()
            break

        screen = capture_screen()

        # Find positions of falling items (beer bottles)
        item_positions = find_items(screen, template, threshold=matching_threshold)
        if not item_positions:
            print("No items detected.")
            current_target_x = None
            continue

        # Target the closest item (lowest y-coordinate in the upper half)
        closest_item_x = item_positions[0][0]

        # Only switch to a new item if we have no current target or the closest item is significantly different
        if current_target_x is None or abs(closest_item_x - current_target_x) > 20:
            current_target_x = closest_item_x

        # Move the basket to align with the current target item using absolute positioning
        move_to_item_absolute(current_target_x)
        
        # Maintain focus on this target for a short duration to avoid switching prematurely
        time.sleep(target_lock_duration)

if __name__ == "__main__":
    main()
