import cv2

# Read both images as grayscale (0 = grayscale mode) and resize them to 300x300
img1 = cv2.resize(cv2.imread("images/img3.jpeg", 0), (300, 300))
img2 = cv2.resize(cv2.imread("images/img4.jpg", 0), (300, 300))

# Create ORB detector (ORB = Oriented FAST and Rotated BRIEF)
# It detects keypoints (distinctive image points) and computes descriptors (vectors describing local patches)
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for the first image
# The second argument (None) is a mask to specify which region to process; None = process the whole image
kp1, des1 = orb.detectAndCompute(img1, None)

# Detect keypoints and descriptors for the second image
kp2, des2 = orb.detectAndCompute(img2, None)

# Optionally visualize the keypoints overlaid on the images
imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)

# # the descriptors are the array of numbers
# print(des1.shape)
# print(des2.shape)
# print(des2[1])


# Create a Brute-Force matcher to compare descriptors
bf = cv2.BFMatcher()

# Find the 2 nearest matches for each descriptor in des1 against des2
matches = bf.knnMatch(des1, des2, k=2)  # k=2 for 2 best matches for each descriptor
print(matches)

# List to store good matches passing Lowe's ratio test
good_matches = []

# Apply ratio test: keep only matches where the best match is significantly better than the second-best
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        # Append as a list of one DMatch object because drawMatchesKnn expects a list of lists
        good_matches.append([m])

# Print the number of good matches
print("Number of good matches:", len(good_matches))


# # Optionally show keypoints without matching lines
# cv2.imshow("kp1", imgKp1)
# cv2.imshow("kp2", imgKp2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Draw matches between the two images
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

cv2.imshow("ORB Feature Matching", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
