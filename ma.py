import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the depth stream explicitly
# For the D435i, 640x480 at 30fps is a standard, highly stable resolution
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create a colorizer object to convert raw depth arrays into a human-readable color spectrum
colorizer = rs.colorizer()

print("Streaming depth map. Press 'q' on the video window to quit.")

try:
    while True:
        # Wait for the next available frame set
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Option A: Visualizing using RealSense's native colorizer helper
        # This maps depth values to a color gradient (usually blue-to-red jet map)
        colorized_depth = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized_depth.get_data())

        # Option B (Alternative manual approach): 
        # If you wanted raw grayscale instead, you would uncomment the lines below:
        # raw_depth_matrix = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(raw_depth_matrix, alpha=0.03)
        # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

        # Render the image window
        cv2.imshow("RealSense Depth Map Test", depth_image)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Safely close down the camera pipeline and windows
    pipeline.stop()
    cv2.destroyAllWindows()
