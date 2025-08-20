from PIL import Image, ImageFilter, ImageEnhance
import pickle
import os
import cv2
import numpy as np
import sys

sys.setrecursionlimit(7000)  # increasing recursion count

# Create folders to store input and processed fingerprint images
os.makedirs("./initial_pictures/", exist_ok=True)
os.makedirs("./comparison_pictures/", exist_ok=True)

# Counts how many times the pixel value changes from 0 to 1 around a center pixel
def count_0_1_transitions(neighbours):
    count = 0
    for k in range(len(neighbours)):
        if neighbours[k] == 0 and neighbours[(k + 1) % 8] == 1:
            count += 1
    return count

# Gets the 8 neighbors of a pixel (N, NE, E, SE, S, SW, W, NW)
def find_neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1],image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]]

# Applies adaptive thresholding to improve contrast for fingerprint processing
def adaptive_threshold(pil_img, block_size=15, C=5):
    img = pil_img.convert("L")
    img_array = np.array(img)

    # Pad the image to handle borders
    pad = block_size // 2
    padded_img = np.pad(img_array, pad, mode='reflect')
    binary_array = np.zeros_like(img_array)

    # Go through each pixel and apply local thresholding
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            local_block = padded_img[i:i+block_size, j:j+block_size]
            local_mean = local_block.mean()
            binary_array[i, j] = 255 if img_array[i, j] > local_mean - C else 0

    return Image.fromarray(binary_array.astype(np.uint8))

# Crops, sharpens, and enhances the fingerprint image to isolate ridge edges
def edge_detection(image):
    image = image.crop((580, 180, 700, 400))  # Crop to the finger box
    detecting_img = image.convert("L")
    detected_img = detecting_img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 25))
    detected_img = ImageEnhance.Contrast(detected_img).enhance(2.0)
    detected_img = detected_img.filter(ImageFilter.GaussianBlur(radius=1))
    detected_img = adaptive_threshold(detected_img)
    return detected_img

# Thins out the fingerprint ridges to 1-pixel width using Zhang-Suen algorithm
def zhang_suen_thinning(image):
    image = np.array(image)
    new_array = (image == 0).astype(np.uint8)
    changed = True

    while changed:
        changed = False
        deleting_pixel = []

        # First pass
        for i in range(1, new_array.shape[0] - 1):
            for j in range(1, new_array.shape[1] - 1):
                p1 = new_array[i][j]
                if p1 == 1:
                    neighbours = find_neighbours(i, j, new_array)
                    n_sum = sum(neighbours)
                    transitions = count_0_1_transitions(neighbours)
                    p2, p4, p6, p8 = new_array[i-1][j], new_array[i][j+1], new_array[i+1][j], new_array[i][j-1]
                    if (2 <= n_sum <= 6 and transitions == 1 and (p2 == 0 or p4 == 0 or p6 == 0) and (p4 == 0 or p6 == 0 or p8 == 0)):
                        deleting_pixel.append((i, j))
        for i, j in deleting_pixel:
            new_array[i][j] = 0
        changed = changed or bool(deleting_pixel)

        # Second pass
        deleting_pixel = []
        for i in range(1, new_array.shape[0] - 1):
            for j in range(1, new_array.shape[1] - 1):
                p1 = new_array[i][j]
                if p1 == 1:
                    neighbours = find_neighbours(i, j, new_array)
                    n_sum = sum(neighbours)
                    transitions = count_0_1_transitions(neighbours)
                    p2, p4, p6, p8 = new_array[i-1][j], new_array[i][j+1], new_array[i+1][j], new_array[i][j-1]
                    if (2 <= n_sum <= 6 and transitions == 1 and (p2 == 0 or p4 == 0 or p8 == 0) and (p2 == 0 or p6 == 0 or p8 == 0)):
                        deleting_pixel.append((i, j))
        for i, j in deleting_pixel:
            new_array[i][j] = 0
        changed = changed or bool(deleting_pixel)
    return new_array

# Detects key fingerprint features (endings and bifurcations)
def minutiae_detection(image):
    minutiae = []
    image = np.array(image)
    new_array = (image == 0).astype(np.uint8)

    for i in range(1, new_array.shape[0] - 1):
        for j in range(1, new_array.shape[1] - 1):
            if new_array[i][j] == 1:
                neighbours = find_neighbours(i, j, new_array)
                transitions = count_0_1_transitions(neighbours)
                if transitions == 1:
                    minutiae.append(("ending", (i, j)))
                elif transitions == 3:
                    minutiae.append(("bifurcation", (i, j)))
    return minutiae

# Full technique to  process the fingerprint from webcam frame
def capture_and_process(frame):
    converting_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(converting_image)
    edge_img = edge_detection(image)
    thinned_array = zhang_suen_thinning(edge_img)
    thinned_img = Image.fromarray((thinned_array * 255).astype(np.uint8))
    minutiae_points = minutiae_detection(thinned_img)

    return minutiae_points

# function to recursively match fingerprints 
def recursive_match(input_coords, stored_coords, tolerance, index=0, count=0):
    if index >= len(input_coords):
        return count

    p1_type, p1_coord = input_coords[index]
    for p2_type, p2_coord in stored_coords:
        if are_points_close(p1_coord, p2_coord, tolerance) and p1_type == p2_type:
            count += 1
            break

    return recursive_match(input_coords, stored_coords, tolerance, index + 1, count)

# Checks if two points are within a set distance (tolerance)
def are_points_close(p1, p2, tolerance=2.5):
    return abs(p1[0] - p2[0]) <= tolerance and abs(p1[1] - p2[1]) <= tolerance

# Compares input fingerprint to all stored ones to find a match
def match_fingerprint(minutiae_points, files, tolerance=2.5, match_threshold=0.8, min_matches=5300):
    best_match = None
    best_score = 0

    print("Files considered for matching:")
    for file_path in files:
        print(file_path)

    for file_path in files:
        with open(file_path, "rb") as file:
            stored_minutiae = pickle.load(file)

        input_coords = minutiae_points
        stored_coords = stored_minutiae

        if not input_coords or not stored_coords:
            continue

# Recursively compares each input minutiae point to all stored minutiae points
        match_count = recursive_match(input_coords, stored_coords, tolerance)

        ratio = match_count / max(len(input_coords), len(stored_coords))

        print(f"Matching {os.path.basename(file_path)}: matches= {match_count}, ratio= {ratio:.2f}")

        if match_count >= min_matches and ratio >= match_threshold and ratio > best_score:
            best_score = ratio
            best_match = os.path.basename(file_path).split(".")[0]

    return best_match

# Webcam handling
img = cv2.VideoCapture(0)

while True:
    ret, frame = img.read()
    if not ret:
        print("Failed to grab a frame")
        break
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 1 to login, 2 to signup and ESC to escape",(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Live camera", frame)
    key = cv2.waitKey(1) & 0xFF

    # Login 
    if key == ord("1"):
        while True:
            ret, frame = img.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.flip(frame, 1)

            display_frame = frame.copy()
            cv2.putText(display_frame, "Place your finger in the box and press enter",(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(display_frame, (580, 180), (700, 400), (0, 0, 255), 2)
            cv2.imshow("Live camera", display_frame)

            name_key = cv2.waitKey(1) & 0xFF
            if name_key == 13:
                minutiae_image = capture_and_process(frame)
                if len(minutiae_image) < 50:
                    # Reject low-quality fingerprints
                    cv2.putText(display_frame, "Low-quality fingerprint. Try again.", (100, 450),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Live camera", display_frame)
                    cv2.waitKey(2000)
                    break

                minutiae_files = [os.path.join("./comparison_pictures", file) for file in os.listdir("./comparison_pictures/") if file.endswith(".minutiae.pkl") and not file.startswith(".")]
                match_name = match_fingerprint(minutiae_image, minutiae_files)
                
                if match_name:
                    cv2.putText(display_frame, f"Welcome {match_name}!", (200, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(display_frame, "User not found. Please sign up by pressing 2.", (200, 500),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow("Live camera", display_frame)
                cv2.waitKey(2000)
                break

    # Exit
    elif key == 27:
        break

    # Signup 
    elif key == ord("2"):
        name = ""
        while True:
            ret, frame = img.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Enter your name: {name}", (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Place your finger in the red box", (20, 300),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "then press enter!", (20, 330),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(display_frame, (580, 180), (700, 400), (0, 0, 255), 2)
            cv2.imshow("Live camera", display_frame)

            name_key = cv2.waitKey(1) & 0xFF
            if name_key == 13:
                if name.strip() == "":
                    # Handle invalid name
                    cv2.putText(display_frame, "Invalid name. Try again.", (80, 500),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Live camera", display_frame)
                    cv2.waitKey(2000)
                    continue
                break
            elif name_key == 127:
                name = name[:-1]
            elif 32 <= name_key <= 126:
                name += chr(name_key)

        ret, frame = img.read()
        if ret:
            converting_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(converting_image)
            filename = f"./initial_pictures/{name}.jpg"
            image.save(filename, format="JPEG")
            edge_img = edge_detection(image)
            thinned_array = zhang_suen_thinning(edge_img)
            thinned_img = Image.fromarray((thinned_array * 255).astype(np.uint8))
            thinned_img.save(f"./comparison_pictures/{name}_thinned.jpg")
            minutiae_points = minutiae_detection(thinned_img)

            minutiae_path = f'./comparison_pictures/{name}.minutiae.pkl'

            if os.path.exists(minutiae_path):
                # Handle duplicate registration
                cv2.putText(display_frame, f"A fingerprint for '{name}' already exists.", (80, 500),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Live camera", display_frame)
                cv2.waitKey(3000)
            else:
                with open(minutiae_path, "wb") as file:
                    pickle.dump(minutiae_points, file)
                    print(f"Minutiae data for '{name}' saved successfully.")
# Destroy the window 
img.release()
cv2.destroyAllWindows()
