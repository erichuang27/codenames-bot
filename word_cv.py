import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def read_words(filepath):
    img_board = cv2.imread(filepath)
    img_board = cv2.rotate(img_board, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_board_gray = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)
    img_h,img_w = img_board_gray.shape
    background_thresh = img_board_gray[0][0]
    ADD_THRESH = 90
    blur = cv2.GaussianBlur(img_board_gray,(5,5),0)
    total_thresh = background_thresh + ADD_THRESH
    _,thresh_img = cv2.threshold(blur,total_thresh,255,cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hier = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour,h in zip(contours,hier[0]) if h[3] == -1 and h[2] > -1]
    top_25_contours = sorted(contours, key=lambda x : cv2.contourArea(x) if cv2.contourArea(x) < (img_h * img_w)/25 else 0,reverse=True)[:25]

    # sort x and y later
    coords_and_index = []
    for i,contour in enumerate(top_25_contours):
        x, y, _, _ = cv2.boundingRect(contour)
        coords_and_index.append((i,x,y))
    sorted_y = sorted(coords_and_index,key=lambda x:x[2])
    for i in range(5):
        sorted_y[5 * i:5* (i + 1)] = sorted(sorted_y[5 * i:5* (i + 1)], key=lambda x:x[1])
    top_25_sorted = [top_25_contours[i[0]] for i in sorted_y]


    def flattener(image, pts, w, h):
        """Flattens an image of a card into a top-down 200x300 perspective.
        Returns the flattened, re-sized, grayed image.
        See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
        temp_rect = np.zeros((4,2), dtype = "float32")
        
        s = np.sum(pts, axis = 2)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis = -1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        # Need to create an array listing points in order of
        # [top left, top right, bottom right, bottom left]
        # before doing the perspective transform

        if w <= 0.8*h: # If card is vertically oriented
            temp_rect[0] = tl
            temp_rect[1] = tr
            temp_rect[2] = br
            temp_rect[3] = bl

        if w >= 1.2*h: # If card is horizontally oriented
            temp_rect[0] = bl
            temp_rect[1] = tl
            temp_rect[2] = tr
            temp_rect[3] = br

        # If the card is 'diamond' oriented, a different algorithm
        # has to be used to identify which point is top left, top right
        # bottom left, and bottom right.
        
        if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
            # If furthest left point is higher than furthest right point,
            # card is tilted to the left.
            if pts[1][0][1] <= pts[3][0][1]:
                # If card is titled to the left, approxPolyDP returns points
                # in this order: top right, top left, bottom left, bottom right
                temp_rect[0] = pts[1][0] # Top left
                temp_rect[1] = pts[0][0] # Top right
                temp_rect[2] = pts[3][0] # Bottom right
                temp_rect[3] = pts[2][0] # Bottom left

            # If furthest left point is lower than furthest right point,
            # card is tilted to the right
            if pts[1][0][1] > pts[3][0][1]:
                # If card is titled to the right, approxPolyDP returns points
                # in this order: top left, bottom left, bottom right, top right
                temp_rect[0] = pts[0][0] # Top left
                temp_rect[1] = pts[3][0] # Top right
                temp_rect[2] = pts[2][0] # Bottom right
                temp_rect[3] = pts[1][0] # Bottom left
                
            
        maxWidth = 200
        maxHeight = 300

        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
        M = cv2.getPerspectiveTransform(temp_rect,dst)
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        return warp

    def find_words(top_25_sorted):
        words = []
        for cont in top_25_sorted:
            peri = cv2.arcLength(cont,True)
            approx = cv2.approxPolyDP(cont,0.01*peri,True)
            pts = np.float32(approx)
            corner_pts = pts

            x,y,w,h = cv2.boundingRect(cont)
            width, height = w, h

            average = np.sum(pts, axis=0)/len(pts)
            cent_x = int(average[0][0])
            cent_y = int(average[0][1])
            center = [cent_x, cent_y]

            warp = cv2.rotate(flattener(img_board, pts, w, h),cv2.ROTATE_90_COUNTERCLOCKWISE)
            cropped_img = warp[warp.shape[0]//2 + 20: warp.shape[0]-20, 20:warp.shape[1]-20]
            blur = cv2.GaussianBlur(cropped_img, (3,3), 0)
            contrast = cv2.convertScaleAbs(blur, alpha=1.3, beta=0)
            thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV  + cv2.THRESH_OTSU)[1]
            words.append(pytesseract.image_to_string(thresh, lang='eng', config='--psm 6').strip())
        return words
    words = find_words(top_25_sorted)
    if len(set(words)) == 25:
        print('success')
    else:
        print('fail, not all words may have been found')
    return(words)
    
print(read_words('assets/5x5.jpg'))