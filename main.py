import cv2 
import matplotlib.pyplot as plt
import numpy as np
from create_sift import *
from os import listdir
from os.path import isfile, join

#images = ['images/peething_needle.jpg','images/triskaidekaphile.jpg']
images = ['images/'+f for f in listdir('images/') if isfile(join('images/', f))]
print(images)


sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
image_length= 500

database = get_card_sift(sift,images,image_length)


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break

    # read images
    #img1 = cv2.imread('test.jpg')  
    img1 = frame
    a1,b1,_ = img1.shape
    img1 = cv2.resize(img1,(int(b1*image_length/a1),image_length),interpolation = cv2.INTER_AREA)
    a1,b1,_ = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

    #img2 = cv2.imread() 
    #images = ['images/triskaidekaphile.jpg']
    
    best_img = -1
    best_score = np.Inf
    best_match = -1
    for image in images:

        #feature matching
        descriptors_2 = database[image][1]

        matches = bf.match(descriptors_1,descriptors_2)
        
        matches = sorted(matches, key = lambda x:x.distance)
        score = np.sum([i.distance for i in matches[:50]])
        if score < best_score:
            best_score = score
            best_img = image
            best_match = matches
    if best_score < 55000:
        print("-------------------------")
        print(best_img)
        print("score : " +str(best_score))
        print("-------------------------")

        img2 = cv2.imread(best_img) 
        a2,b2,_ = img2.shape
        img2 = cv2.resize(img2,(int(b2*a1/a2),a1),interpolation = cv2.INTER_AREA)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        keypoints_2 = database[best_img][0]

        img3 = cv2.drawMatches(frame, keypoints_1, img2, keypoints_2, best_match[:50], img2, flags=2)
        
    else:
        img3 = frame
    cv2.imshow("preview",img3)



vc.release()
cv2.destroyWindow("preview")