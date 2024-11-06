import cv2
import pyautogui
import numpy as np
import time
import keyboard  # For 'Esc' key exit

# Define the ROI based on the provided coordinates
x, y, width, height = 1300, 857, 766, 813

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Define a smaller capture height to focus on the upper-to-mid section of the game area for faster processing
capture_height = height // 2  # Capture only the upper half where items are falling

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

    return sorted(positions, key=lambda pos: pos[1])

def move_to_item_absolute(item_x):
    """Moves the basket horizontally to align directly with the predicted item position using absolute positioning."""
    pyautogui.moveTo(x + item_x, y + height - 30)  # Move to item_x in game area near bottom

def main():
    print("Starting AI... Press 'Esc' to stop.")
    time.sleep(2)

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
            continue

        # Target the closest item (lowest y-coordinate in the upper half)
        closest_item_x = item_positions[0][0]
        
        # Move the basket to align with the predicted landing position of the closest item
        move_to_item_absolute(closest_item_x)

        # Minimal sleep for maximum responsiveness
        time.sleep(0.005)  # Adjust as needed for CPU load; can be reduced further or removed

if __name__ == "__main__":
    main()
