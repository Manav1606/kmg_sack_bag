import cv2
import json
import os

# Global variables
points = []
frame = None

def click_event(event, x, y, flags, param):
    """Handles mouse click events."""
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            height, width, _ = frame.shape
            print(f"Resolution: {width}x{height}, Coordinates: {points}")
            cv2.imshow("Frame", frame)

def save_coordinates_to_config(config_path, camera_id, coordinates):
    """Save coordinates to the config file."""
    with open(config_path, 'r') as file:
        config = json.load(file)

    config['loi'][camera_id] = str(coordinates)  # Save as string format

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    print(f"Coordinates saved for camera '{camera_id}': {coordinates}")

def draw_line_of_interest(rtsp_link, resize_frame=False, resize_width=None, resize_height=None):
    """Open video, capture frame, and allow user to draw line."""
    global frame, points
    points = []

    cap = cv2.VideoCapture(rtsp_link)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Cannot read frame.")
        return None

    # Resize frame if needed
    if resize_frame and resize_width and resize_height:
        frame = cv2.resize(frame, (resize_width, resize_height))

    # Print resolution
    height, width, _ = frame.shape
    print(f"Resolution: {width}x{height}")

    # Display frame
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)

    # Wait for 'q' key
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q') and len(points) == 2:
            break

    cv2.destroyAllWindows()
    return points if len(points) == 2 else None

def main():
    config_path = 'config.json'  # Config file path
    if not os.path.exists(config_path):
        print("Error: Configuration file not found.")
        return

    # Read config file
    with open(config_path, 'r') as file:
        config = json.load(file)

    while True:
        # Display cameras
        print("\nAvailable Cameras:")
        for camera_id, rtsp_link in config['rtsp'].items():
            print(f"Camera ID: {camera_id}, RTSP: {rtsp_link}")

        # Ask user to select camera
        camera_id = input("\nEnter the Camera ID to draw line of interest (or type 'exit' to quit): ").strip()
        if camera_id.lower() == 'exit':
            break

        if camera_id not in config['rtsp']:
            print("Invalid Camera ID. Please try again.")
            continue

        rtsp_link = config['rtsp'][camera_id]

        # Ask if the user wants to resize the frame
        resize_option = input("Do you want to resize the frame? (y/n): ").strip().lower()
        resize_frame = False
        resize_width, resize_height = None, None

        if resize_option == 'y':
            try:
                resize_width = int(input("Enter the desired width: "))
                resize_height = int(input("Enter the desired height: "))
                resize_frame = True
            except ValueError:
                print("Invalid input. Proceeding without resizing.")

        # Capture frame and draw line
        coordinates = draw_line_of_interest(rtsp_link, resize_frame, resize_width, resize_height)
        if coordinates:
            save_coordinates_to_config(config_path, camera_id, coordinates)

        # Ask if user wants to continue
        cont = input("\nDo you want to draw line of interest for another camera? (yes/no): ").strip().lower()
        if cont != 'yes':
            break

if __name__ == "__main__":
    main()
