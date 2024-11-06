import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit

# Define the ROI based on the provided coordinates
x, y, width, height = 1200, 857, 766, 813

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Define a smaller capture height to focus on the upper-to-mid section of the game area for faster processing
capture_height = height // 2  # Capture only the upper half where items are falling

# Drop threshold y-coordinate (items below this y cannot be collected)
drop_threshold_y = 1710 - y  # Set based on the observation you provided

# Initialize variables to track the currently targeted item
current_target_x = None
target_lock_duration = 0.1  # Time to stay focused on a target (in seconds)

def capture_screen():
    """Captures a screenshot of the focused ROI area and returns it in grayscale."""
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

def find_items(screen, template, threshold=0.6, debug=False):
    """Finds all positions of items in the given screen and returns sorted positions by y-coordinate."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    positions = [(loc[0] + template.shape[1] // 2, loc[1] + template.shape[0] // 2) for loc in zip(*locations[::-1])]
    
    # Debugging visuals to see matched items
    if debug:
        for loc in zip(*locations[::-1]):
            top_left = (loc[0], loc[1])
            bottom_right = (loc[0] + template.shape[1], loc[1] + template.shape[0])
            cv2.rectangle(screen, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow("Match Debug", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close debugging window
            cv2.destroyAllWindows()

    # Filter out items below the drop threshold
    return sorted([pos for pos in positions if pos[1] < drop_threshold_y], key=lambda pos: pos[1])

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the target item using absolute positioning."""
    pyautogui.moveTo(x + item_x, y + height - 30)  # Move to item_x in game area near bottom

def main():
    global current_target_x  # Track the current target
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)

    screen_center_x = x + width // 2  # Approximate center of the game area horizontally

    while True:
        if keyboard.is_pressed("esc"):
            print("Exiting...")
            cv2.destroyAllWindows()
            break

        screen = capture_screen()

        # Find positions of falling items (beer bottles)
        item_positions = find_items(screen, template, threshold=0.6)
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
