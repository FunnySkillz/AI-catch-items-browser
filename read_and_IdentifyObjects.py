import cv2
import pyautogui
import numpy as np
import logging

# Configure logging to save to "game_log.txt"
logging.basicConfig(filename="game_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting AI debug log for screen capture and item detection")

# Define the Region of Interest (ROI) for the game screen capture
x, y = 1000, 450  # Starting coordinates of the ROI
width = 1400      # Width of the ROI
height = 1220     # Height of the ROI
capture_height = height  # Set capture height to cover the full vertical game area

# Load the template image for the item (e.g., a beer bottle)
# Ensure the template image (beer_bottle_template.png) is in the same directory as this script
template = cv2.imread("beer_bottle_template.png", cv2.IMREAD_GRAYSCALE)
if template is None:
    logging.error("Template image not found. Make sure 'beer_bottle_template.png' is in the script directory.")
else:
    template_width, template_height = template.shape[::-1]
    logging.info(f"Template loaded with dimensions: {template_width}x{template_height}")

def capture_screen():
    """
    Captures a screenshot of the specified Region of Interest (ROI) area and returns it in grayscale.
    
    Returns:
        numpy.ndarray: Grayscale image of the captured screen region.
    """
    logging.info("Capturing the screen within the defined ROI.")
    # Capture the screen within the defined ROI
    screenshot = pyautogui.screenshot(region=(x, y, width, capture_height))
    # Convert the captured image to BGR color format for OpenCV
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    # Convert the image to grayscale for easier template matching
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    logging.info("Screen captured and converted to grayscale.")
    return gray_screen

def find_items(screen, template, threshold=0.6):
    """
    Finds all positions of items in the given screen based on the provided template.
    
    Args:
        screen (numpy.ndarray): Grayscale image of the current game screen.
        template (numpy.ndarray): Grayscale template image of the item to detect.
        threshold (float): Matching threshold (default 0.6) for detecting items.
        
    Returns:
        list of tuple: Sorted list of detected item positions, each position is a tuple (x, y).
    """
    logging.info("Starting template matching to detect items.")
    # Perform template matching to detect the item
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    # Get the coordinates of locations with high confidence matches
    locations = np.where(result >= threshold)
    
    # Calculate the center coordinates of each detected item
    positions = [(loc[0] + template_width // 2, loc[1] + template_height // 2) for loc in zip(*locations[::-1])]
    logging.info(f"Detected {len(positions)} items with threshold >= {threshold}.")
    
    # Return the positions sorted by y-coordinate (from top to bottom of screen)
    sorted_positions = sorted(positions, key=lambda pos: pos[1])
    logging.debug(f"Sorted item positions by y-coordinate: {sorted_positions}")
    return sorted_positions

# Test these functions independently to see if items are detected
if __name__ == "__main__":
    logging.info("Starting main testing section.")
    # Capture the game screen
    screen = capture_screen()
    
    # Display the screen to verify capture (comment out in headless environments)
    cv2.imshow("Captured Screen", screen)
    cv2.waitKey(500)  # Show screen for 500ms
    logging.info("Displayed captured screen for visual verification.")
    
    # Find items on the captured screen
    detected_items = find_items(screen, template)
    
    # Print detected items' positions for verification
    print("Detected item positions:", detected_items)
    logging.info(f"Detected item positions: {detected_items}")
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    logging.info("Closed all OpenCV windows.")
