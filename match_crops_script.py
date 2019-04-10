import cv2 
import matplotlib.pyplot as plt
import numpy as np
import collections
import time
import glob
import hashlib
import argparse
import sys

def md5(fname):
    ''' To calculate md5 checksum of images '''

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def getUniqueCrops(crops_path):
    ''' To combine repeated crops in a hashed dict file, so that it wont repeat while matching '''

    unique = {}
    for fname in glob.glob(crops_path):
        checksum = md5(fname)
        if checksum in unique:
            unique[checksum].append(fname)
        else:
            unique[checksum] = [fname]
            
    return unique

def matchCrop(img,crop,threshold):
    ''' To match image with the cropped template. Returns -1 if no match found '''

    if threshold > 1.0 or threshold < 0.0:
        return -1    
    
    img = cv2.GaussianBlur(img,(5,5),0)
    crop = cv2.GaussianBlur(crop,(5,5),0)
    
    ih,iw = img.shape[:2]
    h,w = crop.shape[:2]
    
    if ih//h > 8 or iw//w > 8:
        return -1
    
    q = collections.deque([[-2,0,0],[-2,0,0]],2)

    if iw/ih > w/h :
        crop = cv2.resize(crop,(int(w*ih/h),ih))
    else:
        crop = cv2.resize(crop,(iw,int(h*iw/w)))

    for scale in range(20):

        h,w = crop.shape[:2]

        if ih//h > 8 or iw//w > 8:
            break

        res = cv2.matchTemplate(img,crop,cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        bottom_right = (max_loc[0]+w , max_loc[1]+h)

        if max_val > q[-1][0]:
            if max_val > threshold:
                q.append([max_val,max_loc,bottom_right])
        else:
            break

        h = int(h*0.86)
        w = int(w*0.86)
        crop = cv2.resize(crop,(w,h))
        
    if q[-1][0] > threshold:
        return q[-1]
    else:
        return -1

def getCropsAssociation(im_path,keys,found_crops,threshold):
    ''' Helper function to pass images and crops to matcher function '''
    
    matches = []
    
    im = cv2.imread(im_path)

    for key in keys:
        cr = cv2.imread(unique_crops[key][0])
        result = matchCrop(im,cr,threshold)

        if result != -1:
            max_val,top_left,bottom_right = result
            
            if key not in found_crops:
                found_crops.append(key)
                
            for val in unique_crops[key]:
                crop_name = val.split('/')[1]
                matches.append((crop_name,[top_left[0],top_left[1],bottom_right[0],bottom_right[1]]))
    
    return matches

if __name__ == "__main__":
    
    # Assuming crops and image folders are in the same parent folder as this python script
    CROPS_PATH = 'crops'
    IMAGES_PATH = 'images'
    
    if CROPS_PATH[:-1] != '/':
        CROPS_PATH+='/'
    if IMAGES_PATH[:-1] != '/':
        IMAGES_PATH+='/'
        
    dict_file = "dict_file.txt"
    threshold = 0.8
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold')
    parser.add_argument('--dictfile')
    args = parser.parse_args()
    
    print()
    
    if args.threshold:
        threshold = float(args.threshold)
        if threshold > 1.0 or threshold < 0 :
            print("Threshold value should be between 0.0 and 1.0")
            sys.exit(0) 

    if args.dictfile:
        dict_file = args.dictfile   
        
    print("File to save output: ",args.dictfile)
    print("Threshold given: ",args.threshold)
    
    found_crops = []
    not_found = []
    match_dict = {}
    
    unique_crops = getUniqueCrops(CROPS_PATH+'*')
    keys = unique_crops.keys()

    t1 = time.time()

    images = [path.split('/')[1] for path in glob.glob(IMAGES_PATH+'*')]
    
    if (len(images)==0):
        print("Images directory incorrect\n")
        sys.exit(0) 
    
    if (len(keys)==0):
        print("Crops directory incorrect\n")
        sys.exit(0) 
    
    print("\n---------------- Started scanning -----------------\n")
    
    
    ### Magic happens here ! ###
    for image_name in images[:1]:
        print("Scanning matches for image "+image_name)
        match_dict[image_name] = getCropsAssociation(IMAGES_PATH+image_name,keys,found_crops,threshold)

    for key in keys:
        if key not in found_crops:
            cr_names = []
            for path in unique_crops[key]:
                not_found.append((path.split('/')[1],[]))
                
    match_dict['na'] = not_found

    t2 = time.time()

    print("Time taken to get matches: ",t2-t1," secs")
    print("\n----------------------------------------------------")
    print("writing JSON dictionary to \""+dict_file+"\"")
    
    # write to file
    with open(dict_file, 'w') as f:
        f.write('{')
        for key, value in match_dict.items():
            f.write('\'%s\':%s,\n' % (key, value))
        f.write('}')
    print("write finished")
    print("----------------------------------------------------")

