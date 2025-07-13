import os
import cv2

# path to images folder
path = "images"

# loaded images list
loaded_images = []

# names of images without extension
classNames = []

# saved images list
images_list = os.listdir(path)

# Create ORB detector and descriptor extractor
orb = cv2.ORB_create()


# load all the saved images from the folder
def load_folder_images():
    for img in images_list:
        current_img = cv2.imread(f"{path}/{img}", 0)
        loaded_images.append(current_img)
        classNames.append(os.path.splitext(img)[0])
    print(loaded_images)


load_folder_images()


# saving descriptors of all images in a list
def find_descriptors():
    descriptor_list = []
    for img in loaded_images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        descriptor_list.append(descriptors)
    print(descriptor_list)
    return descriptor_list


descriptorList = find_descriptors()


def find_id(img, des_list):
    # Compute keypoints and descriptors for the test image
    kp2, des2 = orb.detectAndCompute(img, None)

    # Create a Brute-Force matcher for descriptor comparison
    bf = cv2.BFMatcher()

    # List to store the number of good matches for each image in des_list
    match_list = []

    # Initialize the index of the best matching image
    best_match_index = -1
    try:
        # Loop over each set of descriptors from the known images
        for des in des_list:
            # Find the 2 nearest matches for each descriptor in des2
            matches = bf.knnMatch(des, des2, k=2)
            good_matches = []

            # Apply Lowe's ratio test to filter out ambiguous matches
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    # Append as a list of one DMatch (required by drawMatchesKnn)
                    good_matches.append([m])

            # Store the count of good matches for this image
            match_list.append(len(good_matches))
    except:
        # If any error occurs (e.g., descriptors are None), ignore and continue
        pass

    # If we found any matches
    if len(match_list) > 0:
        # If the best match has more than 5 good matches (threshold for reliability)
        if max(match_list) > 5:
            # Get the index of the image with the most good matches
            best_match_index = match_list.index(max(match_list))

    # Return the index of the best matching image (-1 if no good match found)
    return best_match_index


# function to test detector with the testing image
def test_detector():
    # Load the test image as grayscale
    testing_img = cv2.imread("images/img4.jpg", cv2.IMREAD_GRAYSCALE)

    # Find the index of the best matching image in the dataset using descriptors
    best_matching_index = find_id(testing_img, descriptorList)

    # If a good match found
    if best_matching_index != -1:
        print("best_matching_index", best_matching_index)
        print(images_list[best_matching_index])

        # Load the matching image from dataset and resize it for display
        best_matching_img = cv2.resize(
            cv2.imread(f"{path}/{images_list[best_matching_index]}"), (300, 300)
        )

        # Show the matching image in a window
        cv2.imshow("Matching Image", best_matching_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Matching Image")
    else:
        # If no match found, print a message
        print("The matches are not found for your image")


test_detector()
