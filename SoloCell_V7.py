import cv2
import numpy as np
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tqdm import tqdm
from PIL import Image, ImageEnhance
from PIL import Image
from io import BytesIO
from datetime import datetime
from tensorflow.keras.models import load_model
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
Image.MAX_IMAGE_PIXELS = None
results_data=[]
confidences=[]


def select_folder():
    # Create a root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window as we only need the file dialog
    
    # Open the folder selection dialog
    folder_path = filedialog.askdirectory()
    
    # Print the selected folder path
    print("Selected result folder: "+folder_path)
    
    # It's important to destroy the root window after use
    root.destroy()
    return folder_path


def log(text):
    global result_folder_path
    print(text)
    with open(result_folder_path+"/log.txt","a") as logfile:
        logfile.write(str(datetime.now().strftime("%m/%d/%y %H_%M_%S"))+": "+text+"\n")

def select_keras_model():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select AI model", filetypes=[
            ("AI file", "*.keras"),  # Specify image file types
 # TIFF files
        ]
    )
    root.destroy()
    return file_path

def select_file_high_res():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select hig res image", filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),  # Specify image file types
            ("JPEG", "*.jpg;*.jpeg"),  # JPEG files
            ("PNG", "*.png"),  # PNG files
            ("GIF", "*.gif"),  # GIF files
            ("BMP", "*.bmp"),  # BMP files
            ("TIFF", "*.tiff")  # TIFF files
        ]
    )
    root.destroy()
    return file_path


def distance_of_cells(point1, point2):
    # Assuming point1 and point2 are tuples with the structure: (data, (x, y), number)
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
def select_roi(org_image):
    org_height=org_image.shape[0]
    org_width=org_image.shape[1]
    roi_width=1600
    scale=org_width/roi_width
    roi_height=round(org_image.shape[0]/scale)
    resized_image = cv2.resize(org_image, (roi_width, roi_height))
    roi = cv2.selectROI(resized_image)
    x, y, width, height = roi
    cv2.rectangle(resized_image, (x,y), (x+width,y+height), (0, 255, 0), 3)
    log("Selected ROI coordinates in resized image (x1,y1,x2,y2): "+str(x)+","+str(y)+","+str(x+width)+","+str(y+height))
    log("Scale during ROI selection: "+str(scale))
    org_x=x*scale
    org_y=y*scale
    org_width=width*scale
    org_height=height*scale
    cv2.destroyAllWindows()
    log("Selected ROI coordinates in original image (x1,y1,x2,y2) will be converted to integer: "+str(org_x)+","+str(org_y)+","+str(org_x+org_width)+","+str(org_y+org_height))
    return org_x, org_y, org_width, org_height
   
       
def calculate_slope(x1, y1, x2, y2):
    # Calculate the slope
   
    if x2 != x1:
        slope= (y2 - y1) / (x2 - x1)
    else:
        slope=  0.0000001 # Undefined slope
    angle = math.degrees(math.atan(slope))
    return angle, slope
def rotate_image(image, angle,interpolation=cv2.INTER_NEAREST):
    # Get the image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w / 2, h / 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
    return rotated

     
def preprocess_frame_for_CNN(image_array):
    """
    Preprocess a single image: read, invert colors, resize, and normalize.
   
    Parameters:
    - image_array: The image as a NumPy array.
    Returns:
    - img: Preprocessed image ready for prediction.
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        grayscale_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_img = image_array
   
    inverted_img = cv2.bitwise_not(grayscale_img)
    resized_img = cv2.resize(inverted_img, (70, 70))  # Resize the image
    normalized_img = resized_img / 255.0  # Normalize pixel values to the range [0, 1]
    img = normalized_img.reshape(-1, 70, 70, 1)  # Add batch dimension for the model
    return img


def rotate_centers_back(coordinate,center,angle):
    coordinate_angle = -angle  # Rotate the coordinate by 45 degrees

    # Get the rotation matrix for the coordinate using the center and angle
    coordinate_rotation_matrix = cv2.getRotationMatrix2D(center, coordinate_angle, 1.0)
   
    # Convert the coordinate to homogeneous coordinates (x, y, 1)
    coordinate_homogeneous = np.array([coordinate[0], coordinate[1], 1])
   
    # Apply the rotation matrix to the coordinate
    rotated_coordinate = np.dot(coordinate_rotation_matrix, coordinate_homogeneous)
   
    # Extract the rotated (x, y) coordinate
    rotated_x, rotated_y = rotated_coordinate[:2]
    rotated_coordinate=(round(rotated_x), round(rotated_y))
    return rotated_coordinate


def save_excel(df, filename):
    global result_folder_path
    """
    Save the dataframe to an Excel file with a unique name by appending a number if necessary.

    Args:
    df : pandas.DataFrame
        The DataFrame to save to an Excel file.
    filename : str
        The base filename to save the DataFrame, without extension.

    Returns:
    None
    """
    # Initialize the counter and set the extension
    counter = 1
    file_extension = '.xlsx'
    new_filename = filename + file_extension

    # Check if the file exists and update the filename with a counter until it doesn't exist
    while os.path.exists(new_filename):
        new_filename = f"{filename}_{counter}{file_extension}"
        counter += 1

    # Save the DataFrame to the new unique filename
    new_filename=result_folder_path+"/"+new_filename
    
    df.to_excel(new_filename, index=False)
    log("AI result for frames saved: "+new_filename)


def calculate_total_distance(df, coord_column):
    # Calculate distances between consecutive coordinates
    distances = [euclidean(df[coord_column].iloc[i], df[coord_column].iloc[i+1]) 
                 for i in range(len(df[coord_column])-1)]
    total_distance = sum(distances)
    return df, total_distance


def reorder_by_nearest_neighbor(df, coord_column):
    # Extract coordinates into a numpy array
    coords = np.array(df[coord_column].tolist())  # Adjust this if the coordinates need conversion

    # Initialize NearestNeighbors with enough neighbors to ensure we don't run out of options
    nbrs = NearestNeighbors(n_neighbors=len(df), algorithm='auto').fit(coords)
    
    # Initialize path, total distance and tracking variables
    path = [0]  # Start at the first index, could be random
    total_distance = 0
    current_index = 0
    visited = set(path)

    while len(visited) < len(df):
        # Query the nearest neighbors of the current point
        distances, indices = nbrs.kneighbors([coords[current_index]])

        # Find the nearest unvisited point
        found = False
        for dist, idx in zip(distances[0], indices[0]):
            if idx not in visited:
                visited.add(idx)
                path.append(idx)
                total_distance += dist
                current_index = idx
                found = True
                break

        if not found:  # If no unvisited neighbors are found, break the loop
            break
    
    # Reorder the dataframe according to the path found
    reordered_df = df.iloc[path].reset_index(drop=True)
    
    return reordered_df, total_distance


def cluster_and_order_points(df, coord_column):
    n_clusters=df.shape[0]//500
    n_clusters=max([1, n_clusters])
    # Extract coordinates into a numpy array
    coords = np.array(df[coord_column].tolist())
    
    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    df['cluster'] = kmeans.labels_

    # This will hold the final path dataframes
    full_path = []

    # Process each cluster
    for cluster in range(n_clusters):
        cluster_df = df[df['cluster'] == cluster]
        if cluster_df.empty:
            continue
        
        # Extract cluster points
        cluster_coords = np.array(cluster_df[coord_column].tolist())
        nbrs = NearestNeighbors(n_neighbors=len(cluster_coords), algorithm='auto').fit(cluster_coords)
        
        # Initialize path and variables for the cluster
        path = [0]
        current_index = 0
        visited = set(path)

        while len(visited) < len(cluster_coords):
            _, indices = nbrs.kneighbors([cluster_coords[current_index]])
            for idx in indices[0]:
                if idx not in visited:
                    visited.add(idx)
                    path.append(idx)
                    current_index = idx
                    break

        # Reorder the cluster DataFrame according to the path and append to full path
        reordered_cluster_df = cluster_df.iloc[path].reset_index(drop=True)
        full_path.append(reordered_cluster_df)
    
    # Combine all cluster paths into one DataFrame
    final_df = pd.concat(full_path).reset_index(drop=True)

    # Calculate the total distance from the final ordered DataFrame
    none, total_distance = calculate_total_distance(final_df, coord_column)

    return final_df, total_distance

def get_cell_frames(rotated_image, row_seed, column_seed, square_size, square_distance, skipper,laser_frame_size):
    global roi_frame, angle, results_data, confidences, org_image
   
    column_border1, row_border1= (roi_frame[0],roi_frame[1])
    column_border2, row_border2= (column_border1+roi_frame[2],row_border1+roi_frame[3])
    column_border1, row_border1= round(column_border1), round(row_border1)
    column_border2, row_border2= round(column_border2), round(row_border2)

    column = column_seed-round(square_size/2)
    row = row_seed-round(square_size/2)
   
    new_columns_right=[]
    while column<column_border2:
        frame=rotated_image[row:row+square_size,column:column+square_size]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([179, 255, 240])
        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
             
        if contours:
            # Calculate moments for the largest contour
            M = cv2.moments(contours[0])

            # Calculate the centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = round(square_size/2), round(square_size/2)

               
            result=round(cX-square_size/2)
            if abs(result)>square_distance-square_size:
                if result<0:
                    result=-(square_distance-square_size)
                else:
                    result=+(square_distance-square_size)
           
            column=column+result
            new_columns_right.append(column)
            
        else:
            column=new_columns_right[-1]+square_distance
            new_columns_right.append(column)
        
           
        column+=square_distance
       
    column = column_seed-round(square_size/2)-square_distance
    new_columns_left=[]
    while column>column_border1:
        frame=rotated_image[row:row+square_size,column:column+square_size]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([179, 255, 240])
        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        if contours:
            # Calculate moments for the largest contour
            M = cv2.moments(contours[0])

            # Calculate the centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = round(square_size/2), round(square_size/2)
               
               
            result=round(cX-square_size/2)
            if abs(result)>square_distance-square_size:
                if result<0:
                    result=-(square_distance-square_size)
                else:
                    result=+(square_distance-square_size)
                   
            column=column+result
            new_columns_left.append(column)
            
        else:
            column=new_columns_left[-1]-square_distance
            new_columns_left.append(column)
           
        column-=square_distance
    new_columns_left.reverse()
    columns=new_columns_left+new_columns_right
    new_rows_bottom=[]
    row = row_seed-round(square_size/2)
    column = new_columns_right[0]
    la=0
    while row<row_border2:
        frame=rotated_image[row:row+square_size,column:column+square_size]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([179, 255, 240])
        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

#         cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        if contours:
            # Calculate moments for the largest contour
            M = cv2.moments(contours[0])

            # Calculate the centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
               
            else:
               
                cX, cY = round(square_size/2), round(square_size/2)
               
            result=round(cY-square_size/2)
            if abs(result)>square_distance-square_size:
                if result<0:
                    result=-(square_distance-square_size)
                else:
                    result=+(square_distance-square_size)

            row=row+result
            new_rows_bottom.append(row)

        else:
            row=new_rows_bottom[-1]+square_distance
            new_rows_bottom.append(row)
       
        row+=square_distance

       
       
    row = row_seed-round(square_size/2)-square_distance
    column = new_columns_right[0]
    new_rows_upper=[]
    while row>row_border1:
        frame=rotated_image[row:row+square_size,column:column+square_size]
       
           
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([179, 255, 240])
        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        if contours:
            # Calculate moments for the largest contour
            M = cv2.moments(contours[0])

            # Calculate the centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = round(square_size/2), round(square_size/2)
               
               
            result=round(cY-square_size/2)
            if abs(result)>square_distance-square_size:
                if result<0:
                    result=-(square_distance-square_size)
                else:
                    result=+(square_distance-square_size)
                   
            row=row+result
            new_rows_upper.append(row)
        else:
            row=new_rows_upper[-1]-square_distance
            new_rows_upper.append(row)
            

           
        row-=square_distance
    new_rows_upper.reverse()
    rows=new_rows_upper+new_rows_bottom
       

    count=0
    no_contour=0
    single_cell_frame_corners_list=[]
    complete_black=rotated_image.copy()
    complete_black[:] = (0, 0, 0)
    all_colors=[]
    confidences=[]
    results_data.append(len(rows)*len(columns))
    rotated_coordinates=[]
    count_row=0
    for row in tqdm(rows):
        count_row+=1
        count_column=0
        for column in columns:
            count_column+=1
            count+=1
            if rotated_image[row:row+square_size,column:column+square_size].min()<170:
                frame_for_draw=rotated_image[row:row+square_size,column:column+square_size]
                frame_for_pred=preprocess_frame_for_CNN(rotated_image[row:row+square_size,column:column+square_size])
                # cv2.rectangle(rotated_image, (column,row), (column+square_size,row+square_size), (255, 0, 0), 1)
                confidence=model.predict(frame_for_pred,verbose=0)[0]
                   
                # cv2.rectangle(rotated_image, (column,row), (column+square_size,row+square_size), (255, 0, 0), 1)
                hsv_roi = cv2.cvtColor(frame_for_draw, cv2.COLOR_BGR2HSV)
                lower_hsv = np.array([0, 0, 0])
                upper_hsv = np.array([179, 255, 245])
                mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
                     
                if contours:
                    
                    # Calculate moments for the largest contour
                    M = cv2.moments(contours[0])
                   
                    # Calculate the centroid
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = column+square_size/2, row+square_size/2

                    confidences.append([count,[count_row, count_column], [column + cX , row + cY ], confidence])
                else:
                    no_contour+=1
        columns.reverse()


               
# FOR LABELING        
#             cv2.namedWindow("Decide number", cv2.WINDOW_NORMAL)
#             cv2.imshow("Decide number", rotated_image[row:row+square_size,column:column+square_size])
#             cv2.resizeWindow("Decide number", 800, 800)
#             save_frame(rotated_image[row:row+square_size,column:column+square_size],count)
# #             cv2.rectangle(rotated_image, (column,row), (column+square_size,row+square_size), (255, 0, 0), 1)
#             cv2.destroyAllWindows()
#             count+=1
           
           
           
    return confidences
   

####################################################################################################################################

print("Select Result Folder.")
output_folder = select_folder()

print("Select High Resolution Image.")
high_res_file_path = select_file_high_res()

result_folder_path=output_folder+"/"+high_res_file_path.split("/")[-1]+"-"+str(datetime.now().strftime("%m_%d_%y-%H_%M_%S"))
os.makedirs(result_folder_path, exist_ok=False)

log("Selected High Resolution Image: "+high_res_file_path)

log("Result folder created successfully.")
log("Result folder:"+result_folder_path)

keras_path=select_keras_model()
log("Selected AI file:"+keras_path)
model = load_model(keras_path)
log("AI module loaded successfully.")


# Load the image and convert to HSV
org_image = cv2.imread(high_res_file_path)
log("Original image read successfully.")
realshape0,realshape1=org_image.shape[0], org_image.shape[1]
log("Original image dimensions (height,width): "+str(org_image.shape[0])+","+str(org_image.shape[1]))
# Resize Image
# resized_image, scale_ratio_x, scale_ratio_y, max_dimension=resize_image(org_image)
# Select ROI
x, y, width, height=select_roi(org_image)
roi_frame=(x, y, width, height)
framex1=int(x)
framex2=int(x+width)
framey1=int(y)
framey2=round(y+height)


# Initial parameters
zoom_level = 100  # Initial zoom percentage
center = (org_image.shape[1] // 2, org_image.shape[0] // 2)  # Center of the image
points = []  # List to store point coordinates
line_live = False  # Status of live line following the mousez
moving_point = None
s_press_count = 0
last_mouse_position = center 
# Circle size calculation
circle_size = round(org_image.shape[1] / 1800)
circle_size = max(1, circle_size)

log("Circle size during reference selection will be "+str(circle_size))


overlay = np.zeros_like(org_image, dtype=np.uint8)+ 255  # Initialize the overlay


def update_overlay():
    global overlay, last_mouse_position, img, offset, s_press_count,zoom_level
    # Recreate overlay to match the current size of img
    overlay = np.zeros_like(img, dtype=np.uint8)  # Clear previous contents and ensure size match

    # Dynamic font size based on the window size
    window_width = cv2.getWindowImageRect("Image")[2]
    font_scale = window_width / zoom_level
    font_scale = max(font_scale, 0.1)  # Ensure a minimum font scale

    # Text position, avoiding edge overlap
    text_x = min(max(10, last_mouse_position[0] + 10), img.shape[1] - 100)
    text_y = min(max(10, last_mouse_position[1] + 20), img.shape[0] - 10)
    text_position = (text_x, text_y)

    # Determine the information text to display
    info_text = "Default Text"
    if s_press_count < 7:
        info_text = f"Adjacent: {7 - s_press_count}"
    elif s_press_count == 7:
        info_text = "Seeding: 1"
    elif s_press_count == 8:
        info_text = f"Farthest cells: {10 - s_press_count}"
    elif s_press_count == 9:
        info_text = f"Farthest cells: {10 - s_press_count}"
    elif s_press_count == 10:
        info_text = f"Press any key to quit"
    
    # Draw the text on the overlay
    cv2.putText(overlay, info_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

    # Blend overlay with the main image and display
    display_image = cv2.addWeighted(img, 1, overlay,  1, 1)
    cv2.imshow("Image", display_image)


def zoom_image(image, center, zoom):
    height, width = image.shape[:2]
    x, y = center

    # Calculate the cropped window dimensions
    new_width = int(width * 100 / zoom)
    new_height = int(height * 100 / zoom)

    # Ensure the crop rectangle stays within the bounds
    x1 = max(0, x - new_width // 2)
    y1 = max(0, y - new_height // 2)
    x2 = min(width, x1 + new_width)
    y2 = min(height, y1 + new_height)

    x1 = x2 - new_width if x2 - new_width < 0 else x1
    y1 = y2 - new_height if y2 - new_height < 0 else y1

    return image[y1:y2, x1:x2], (x1, y1)

def on_trackbar(val):
    global zoom_level
    zoom_level = max(1, val)
    update_view()

def mouse_callback(event, x, y, flags, param):
    global center, last_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_position = (x, y)  # Update last known mouse position
        update_overlay()
    if event == cv2.EVENT_LBUTTONDOWN:  # Move left
        step_size = max(1, img.shape[1] // 6)
        center = (max(0, center[0] - step_size), center[1])
        update_view()
    elif event == cv2.EVENT_RBUTTONDOWN:  # Move right
        step_size = max(1, img.shape[1] // 6)
        center = (min(org_image.shape[1], center[0] + step_size), center[1])
        update_view()
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # Scroll up
            center = (center[0], max(0, center[1] - img.shape[0] // 6))
        else:  # Scroll down
            center = (center[0], min(org_image.shape[0], center[1] + img.shape[0] // 6))
        update_view()

def update_view():
    global img, offset
    img, offset = zoom_image(org_image, center, zoom_level)
    # Draw all points
    for i, point in enumerate(points):
        cv2.circle(img, (point[0] - offset[0], point[1] - offset[1]), circle_size, (25, 13, 0), 2)
    
    # Connect only the first seven points
    for i in range(1, 7):
        if i < len(points):
            cv2.line(img, (points[i-1][0] - offset[0], points[i-1][1] - offset[1]),
                     (points[i][0] - offset[0], points[i][1] - offset[1]), (0, 255, 0), 2)

    # Explicitly connect the 9th to the 10th 's' press
    if len(points) > 9:
        cv2.line(img, (points[8][0] - offset[0], points[8][1] - offset[1]),  # 9th point
                 (points[9][0] - offset[0], points[9][1] - offset[1]), (0, 255, 0), 2)  # 10th point

    cv2.imshow("Image", img)

# Create window and trackbar
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Zoom", "Image", zoom_level, 2500, on_trackbar)
cv2.setMouseCallback("Image", mouse_callback)

# Initial update

# Initial update of the view
update_view()

# Variables for the first 7 's' presses
# Track the last mouse position
# Assume all initialization is already done above

# Variables for the first 7 's' presses
s_press_count = 0
last_mouse_position = center  # Assuming this variable is correctly tracked in your setup

# First while loop for the first 7 's' presses
while s_press_count < 7:
    k = cv2.waitKey(1)
    if k == ord('q'):
        log("Pressed q during reference selection. This exits from the window.")
        break
    elif k == ord('s'):
        x, y = last_mouse_position
        log("Pressed s during reference selection.")
        adjusted_point = (x + offset[0], y + offset[1])
        log("s press position in the original image: "+str(adjusted_point))
        points.append(adjusted_point)
        if len(points) > 1:
            cv2.line(img, (points[-2][0] - offset[0], points[-2][1] - offset[1]),
                     (points[-1][0] - offset[0], points[-1][1] - offset[1]), (0, 255, 0), 2)
        cv2.circle(img, (points[-1][0] - offset[0], points[-1][1] - offset[1]), circle_size, (25, 13, 0), 2)
        log("s press position in the showed window: "+str((points[-1][0] - offset[0], points[-1][1] - offset[1])))

        cv2.imshow("Image", img)
        s_press_count += 1

# Second while loop for the 8th and 9th 's' presses, with no connection between them
while s_press_count >= 7 and s_press_count < 9:
    k = cv2.waitKey(1)
    if k == ord('q'):
        log("Pressed q during reference selection. This exits from the window.")
        break
    elif k == ord('s'):
        log("Pressed s during reference selection.")
        x, y = last_mouse_position
        adjusted_point = (x + offset[0], y + offset[1])
        log("s press position in the original image: "+str(adjusted_point))
        points.append(adjusted_point)
        cv2.circle(img, (adjusted_point[0] - offset[0], adjusted_point[1] - offset[1]), circle_size, (25, 13, 0), 2)
        log("s press position in the showed window: "+str((adjusted_point[0] - offset[0], adjusted_point[1] - offset[1])))
        cv2.imshow("Image", img)
        s_press_count += 1
        if s_press_count == 9:  # Setup for connecting 'd' presses
            last_s_point = adjusted_point  # Save the 9th 's' press for future connection

# Third while loop for 'd' presses and handling the 10th 's' press
while True:
    k = cv2.waitKey(1)
    if k == ord('q'):
        log("Pressed q during reference selection. This exits from the window.")
        break
    elif k == ord('d'):
        log("Pressed d during reference selection.")
        if last_s_point is not None:  # Ensure there is a previous significant point to connect to
            x, y = last_mouse_position
            # Calculate the full image coordinates from the current displayed coordinates
            full_image_x = x + offset[0]
            full_image_y = y + offset[1]
            full_image_point = (full_image_x, full_image_y)
            log("d press position in the original image: "+str(full_image_point))
            # Draw line on the original image (org_image)
            cv2.line(org_image, (last_s_point[0], last_s_point[1]),
                     (full_image_point[0], full_image_point[1]), (255, 0, 0), 2)
            cv2.circle(org_image, (full_image_point[0], full_image_point[1]), circle_size, (25, 13, 0), 2)
            log("d press position in the showed window: "+str((full_image_point[0], full_image_point[1])))
            # Update the last significant 's' press point for the next 'd' press
            last_s_point = full_image_point
            
            # Update the displayed image
            update_view()  # Reflect the change in the display view
    elif k == ord('s') and s_press_count == 9:
        log("Pressed s during reference selection.")
        x, y = last_mouse_position
        s_press_count+=1
        # Convert displayed coordinates to full image coordinates
        adjusted_point = (x + offset[0], y + offset[1])
        log("s press position in the original image: "+str(adjusted_point))
        points.append(adjusted_point)  # Save the point in terms of the original image
        # Connect this 10th 's' press with the last significant 's' (the 9th)
        cv2.line(org_image, (points[8][0], points[8][1]),  # Points already in full image coordinates
                 (adjusted_point[0], adjusted_point[1]), (0, 255, 0), 2)
        cv2.circle(org_image, (adjusted_point[0], adjusted_point[1]), circle_size, (25, 13, 0), 2)
        # Update the displayed image to reflect the changes
        update_view()
        cv2.waitKey(0)  # Wait for any key to close
        break

cv2.destroyAllWindows()
                             
total_distance_of_selected_cells=distance_of_cells([points[0][0],points[0][1]], [points[1][0],points[1][1]])+distance_of_cells([points[1][0],points[1][1]], [points[2][0],points[2][1]])+distance_of_cells([points[2][0],points[2][1]], [points[3][0],points[3][1]])+distance_of_cells([points[3][0],points[3][1]], [points[4][0],points[4][1]])+distance_of_cells([points[4][0],points[4][1]], [points[5][0],points[5][1]])+distance_of_cells([points[5][0],points[5][1]], [points[6][0],points[6][1]])

average_distance=round(total_distance_of_selected_cells/6)
log("Average distance in between cells: "+str(average_distance))
square_distance=average_distance


column_seed, row_seed=points[7][0], points[7][1]


#Calculate angle and rotate 
angle,slope=calculate_slope(points[8][0], points[8][1], points[9][0], points[9][1]) #x1,y1,x2,y2
log("arctan of the angle: "+str(angle))
log("Slope: "+str(slope))
angle=abs(angle)


if slope<0:
    log("slope is negative and angle is "+str(angle))
    if angle>45:
        log("angle is bigger than 45")
        angle=90-angle #turn image this angle clockwise
    else:
        log("angle is smaller than 45")
        angle=-angle #turn image this angle clockwise
else:
    log("slope is positive and angle is "+str(angle))
    if angle>45:
        log("angle is bigger than 45")
        angle=angle-90 #turn image this angle clockwise
    else:
        log("angle is smaller than 45")
        angle=angle#turn image this angle clockwise
    
log("The angle for rotating the image clockwise: "+str(angle))        


# Re-read image for drawing cell framesget_cell
image = Image.open(high_res_file_path)

# Adjust contrast
log("Image pre-processing....")
enhancer = ImageEnhance.Contrast(image)
image_with_adjusted_contrast = enhancer.enhance(5)
log("Image contrasted")
image_with_adjusted_contrast.save(result_folder_path+"/"+'contrasted_image.png')# factor > 1 will increase contrast, factor < 1 will decrease
log("Contrasted image saved for the next step: "+result_folder_path+"/"+'contrasted_image.png')
org_image = cv2.imread(result_folder_path+"/"+'contrasted_image.png')
log("Contrasted image read again for cv2 and preprocessing has been completed.")
# resized_image, scale_ratio_x,scale_ratio_y, max_dimension=resize_image(org_image)
rotated_image=rotate_image(org_image, angle)
log("Original image rotated for frame detections.")
log("Frame detections and AI evaluation will be simultaneously calculated.")
final_data=get_cell_frames(rotated_image, row_seed, column_seed, square_size=round(square_distance*(7/8)), square_distance=square_distance,skipper=0.2, laser_frame_size=round(square_distance*(5/8)))
laser_size=round(square_distance*(5/8))
log("Frame evaluation completed successfully.")


height=realshape0
width=realshape1    
center = (width / 2, height / 2)


confidences=[whole[3][0] for whole in final_data]
# grid_coords=[whole[1] for whole in final_data]
image_coords=[whole[2] for whole in final_data]
df=pd.DataFrame()
# df["Grid Coordinate (row,column)"]=grid_coords


org_image_coordinates=[]
for coor in image_coords:
    image_coordinate=rotate_centers_back(coor,center,angle)
    org_image_coordinates.append(image_coordinate)
    
    
df["Image Coordinate"]=org_image_coordinates
df["Solo Cell Confidence"]=confidences
log("AI result for all frames will be saved.")
save_excel(df,"Solo Cell Result_All_Frames")



log("Preparing Histogram for confidence selection.")
# SHOW HISTOGRAM
fig, ax = plt.subplots()
n, bins, patches = ax.hist(confidences, bins=100, range=(0, 1), edgecolor='black')
ax.set_title('Histogram of Confidence Levels')
ax.set_xlabel('Confidence')
ax.set_ylabel('Frequency')
ax.set_xlim([0, 1])
ax.set_xticks(np.linspace(0, 1, 11))  # Set 11 ticks from 0 to 1 (including 0 and 1)

# Setting non-frame text and ticks to pure black
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.title.set_color('black')
ax.tick_params(axis='both', colors='black')

# Customize frame color for better detection
frame_color = (0, 0.8, 0)  # RGB color for the frame in the 0-1 range (green)
for spine in ax.spines.values():
    spine.set_edgecolor(frame_color)
    spine.set_linewidth(1)

# Save the plot to a buffer
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
buffer.seek(0)

# Convert the buffer to an OpenCV image
image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.resize(image, (800, 600))

# Convert the frame color back to 0-255 range for OpenCV
frame_color_cv2 = tuple(int(255 * c) for c in frame_color)

# Function to find the frame bounds in the image
def find_frame_bounds(img):
    mask = cv2.inRange(img, frame_color_cv2, frame_color_cv2)
    coords = np.where(mask)
    if coords[0].size > 0:
        x_min = np.min(coords[1])
        x_max = np.max(coords[1])
        return x_min, x_max
    return None, None

x_start, x_end = find_frame_bounds(image)

# Handle mouse events
# Handle mouse events
# Handle mouse events
def mouse_event(event, x, y, flags, param):
    global last_confidence, result_folder_path
    if x < x_start or x > x_end:
        return
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        img_copy = image.copy()
        cv2.line(img_copy, (x, 0), (x, img_copy.shape[0]), (0, 255, 0), 1)
        
        x_value = (x - x_start) / (x_end - x_start)
        bin_index = np.digitize([x_value], bins)[0] - 1
        if bin_index < 0 or bin_index >= len(n):
            bin_index = max(0, min(bin_index, len(n) - 1))
        count_above = np.sum(n[bin_index:])
        
        text1 = f'Confidence: {x_value:.3f}'
        text2 = f'Count: {count_above}'
        text_height = 20
        text1_width = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
        text2_width = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
        if x + text1_width + 10 > img_copy.shape[1]:
            cv2.putText(img_copy, text1, (x - text1_width - 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (162,153,12), 1)
            cv2.putText(img_copy, text2, (x - text2_width - 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (162,153,12), 1)
        else:
            cv2.putText(img_copy, text1, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (162,153,12), 1)
            cv2.putText(img_copy, text2, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (162,153,12), 1)
        
        cv2.imshow("Histogram", img_copy)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        last_confidence = round(x_value,3)
        cv2.imwrite(result_folder_path+"/"+"Histogram.png", img_copy)
        cv2.destroyAllWindows()
        log("User selection for confidence level is completed. Selected confidence value is "+str(last_confidence))
        log("The histogram is saved successfully.")
        


cv2.imshow("Histogram", image)
cv2.setMouseCallback("Histogram", mouse_event)
cv2.waitKey(0)
cv2.destroyAllWindows()



df_selected=df[df["Solo Cell Confidence"]>last_confidence]
log("Frames are filtrated by the selected confidence.")



log("Path optimization is in progress...")
snake_df,snake_total=calculate_total_distance(df_selected,"Image Coordinate")
log("Snake Algorithm Total Distance: "+str(snake_total))

NN_df,NN_total=reorder_by_nearest_neighbor(df_selected,"Image Coordinate")
log("Nearest Neighbor Algorithm Total Distance: "+str(NN_total))

CN_df,CN_total=cluster_and_order_points(df_selected,"Image Coordinate")
log("Cluster First Nearest Neighbor Algorithm Total Distance: "+str(CN_total))

total_distances=[snake_total,NN_total,CN_total]
idx_min=total_distances.index(min(total_distances))

if idx_min==0:
    log("Snake Algorithm is the fastest.")
    final_df=snake_df
elif idx_min==1:
    log("Nearest Neighbor Algorithm is the Fastest. "+str(round((NN_total/snake_total),2)*100)+"% Faster Than Snake Algorithm")
    final_df=NN_df
    
elif idx_min==2:
    log("Cluster First Nearest Neighbor Algorithm is the Fastest. "+str(round((CN_total/snake_total),2)*100)+"% Faster Than Snake Algorithm")
    final_df=CN_df



with open(result_folder_path+"/"+"Selected_Coordinates.txt","w") as file:
    for index, coordinate in enumerate(final_df["Image Coordinate"]):
        file.write(str(index+1)+","+str(coordinate[0])+","+str(coordinate[1])+"\n")
        
log("Filtrated and ordered coordinates are saved successfully (number,x,y): "+result_folder_path+"/"+"Selected Coordinates.txt")

log("AI result for selected frames will be saved.")
save_excel(final_df,"Selected_Coordinates")


log("Creating original image with selected frames...")

org_image = cv2.imread(high_res_file_path)
real_coordinates=[]
count=1
font_scale = laser_size/120
log("Font scale for the frame numbers will be "+str(font_scale)+".")
for image_coordinate in final_df["Image Coordinate"]:
            cv2.rectangle(org_image, (round(image_coordinate[0]-laser_size*1/2) , round(image_coordinate[1]-laser_size*1/2)), (round(image_coordinate[0]+laser_size*1/2) , round(image_coordinate[1]+laser_size*1/2)), (0, 255, 0), 1)
            cv2.putText(org_image, str(count), (round(image_coordinate[0] - laser_size * 1/2 - laser_size * 1/8), round(image_coordinate[1] - laser_size * 1/2 - laser_size * 1/8)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
            count+=1


cv2.imwrite(result_folder_path+"/"+"Selected_Frames.png",org_image)
log("Original image with selected coordinates is saved successfully: "+result_folder_path+"/"+"Selected_Frames.png")
log("SoloCell completed successfully!")
