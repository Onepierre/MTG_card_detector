import cv2 
import matplotlib.pyplot as plt
import numpy as np

def get_card_sift(sift,loc_list,image_length):
    dictionnary = {}
    for loc in loc_list:
        img2 = cv2.imread(loc) 
        a2,b2,_ = img2.shape
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        dictionnary[loc] = sift.detectAndCompute(img2,None)
    return dictionnary
