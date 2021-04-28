

def crop_video(video_stack, min_x, min_y, max_x, max_y):
    """
    Opens the provided video (cv2.VideoCapture object) and extracts the frame data
    into a numpy array, cropping each frame at the provided coordinates.
    """
    # Get base height and width
    h_frame = video_stack.shape[1]
    w_frame = video_stack.shape[2]

    # validate new height and width
    h, w = max_y - min_y, max_x - min_x

    if w_frame <= w:
        min_x, max_x = 0, w_frame - 1
        w = w_frame

    if h_frame <= h:
        min_y, max_y = 0, h_frame - 1
        h = h_frame

    # Crop video array
    new_video_stack = video_stack[:, min_y:max_y, min_x:max_x]

    return new_video_stack, w, h, min_x, min_y


def sticker_coord_calibration(one_coords_and_radii, min_x, min_y):

    x_o = one_coords_and_radii[0]
    y_o = one_coords_and_radii[1]

    return tuple([x_o - min_x, y_o - min_y])
