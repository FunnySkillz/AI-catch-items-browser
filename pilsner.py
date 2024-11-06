import cv2
import pyautogui
import numpy as np
import time

# Load the template image for the beer bottle
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
template_width, template_height = template.shape[::-1]

# Load the template image for the basket
basket_template = cv2.imread("basket_template.png", cv2.IMREAD_GRAYSCALE)
basket_template_width, basket_template_height = basket_template.shape[::-1]

def capture_screen():
    # Capture the full screen
    screenshot = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for template matching

def find_positions(screen, template, threshold=0.8):
    # Use template matching to find all positions of the given template in the screen
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)  # Find all locations with confidence above threshold

    positions = []
    for loc in zip(*locations[::-1]):  # Swap x and y locations
        center_x = loc[0] + template.shape[1] // 2
        center_y = loc[1] + template.shape[0] // 2
        positions.append((center_x, center_y))

    # Sort items by vertical position (y-coordinate) for items or return all positions found
    return sorted(positions, key=lambda pos: pos[1]) if positions else None

def main():
    print("Starting AI...")
    time.sleep(2)  # Brief delay to position the game

    while True:
        screen = capture_screen()

        # Find basket position
        basket_position = find_positions(screen, basket_template, threshold=0.8)
        if not basket_position:
            print("Basket not found. Please ensure it's visible on the screen.")
            continue

        # Use the first (and likely only) position of the basket
        basket_x = basket_position[0][0]
        basket_y = basket_position[0][1]

        # Find positions of falling items (beer bottles)
        item_positions = find_positions(screen, template, threshold=0.8)

        # Filter items to focus on the closest one to the basket
        target_item = None
        min_distance = float("inf")
        
        for item_x, item_y in item_positions:
            # Calculate the distance between the item and the basket
            distance_to_basket = abs(item_y - basket_y)
            if distance_to_basket < min_distance:
                target_item = (item_x, item_y)
                min_distance = distance_to_basket

        # If we found an item, move the mouse to align with it
        if target_item:
            target_item_x = target_item[0]
            # Calculate the horizontal offset between the basket and the target item
            offset_x = target_item_x - basket_x

            # Move the mouse only if the item is not already centered with the basket
            if abs(offset_x) > 10:  # Avoid tiny, unnecessary movements
                pyautogui.moveRel(offset_x, 0, duration=0.05)  # Faster movement to align with item

        # Pause briefly to avoid overloading the CPU
        time.sleep(0.05)

if __name__ == "__main__":
    main()
