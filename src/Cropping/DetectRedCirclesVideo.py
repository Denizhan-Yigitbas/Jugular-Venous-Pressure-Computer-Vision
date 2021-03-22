import cv2
import numpy as np

path = "/Users/joshuakowal/Downloads/RedDotVideo.mp4"
cap = cv2.VideoCapture(path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('/Users/joshuakowal/Downloads/ExampleVideo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 60,
                      (frame_width, frame_height))

while True:
    ret, captured_frame = cap.read()

    if ret:
        output_frame = captured_frame.copy()

        # Convert original image to BGR, since Lab is only available from BGR
        captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)
        # First blur to reduce noise prior to color space conversion
        captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)
        # Convert to Lab color space, we only need to check one channel (a-channel) for red here
        captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
        # Threshold the Lab image, keep only the red pixels
        # Possible yellow threshold: [20, 110, 170][255, 140, 215]
        # Possible blue threshold: [20, 115, 70][255, 145, 120]
        captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
        # Second blur to reduce more noise, easier circle detection
        captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
        # Use the Hough transform to detect circles in the image
        circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8,
                                   param1=100, param2=18, minRadius=5, maxRadius=60)

        # If we have extracted a circle, draw an outline
        # We only need to detect one circle here, since there will only be one reference object

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i in range(circles.shape[1]):
                cv2.circle(output_frame, center=(circles[i, 0], circles[i, 1]), radius=circles[i, 2], color=(255, 0, 0),
                           thickness=2)
                cv2.circle(output_frame, center=(circles[i, 0], circles[i, 1]), radius=2, color=(0, 0, 0), thickness=2)

        # Display the resulting frame, quit with q
        out.write(output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Display the resulting frame, quit with q

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
