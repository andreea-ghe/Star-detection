import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt

def increase_brightness(img, threshold=26):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = v > threshold    # increase brightness only for pixels that are brighter, make them equally bright
    v[mask] = np.clip(255, 0, 255)  # Increase brightness, cap at 255

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


class StarDetection:
    
    def __init__(self, image_path):
        self.image_path = image_path

    def keep_only_light_colors(self):
        image = cv2.imread(self.image_path)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV

        lower_light = np.array([0,   0, 100], dtype=np.uint8) # everything that is darker, not star, is label
        upper_light = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_light, upper_light)
        
        only_stars = cv2.bitwise_and(image, image, mask=mask)
        return image, only_stars

    def adjust_image(self):
        image, only_stars = self.keep_only_light_colors()
        
        # gray_blur = cv2.GaussianBlur(only_stars, (5, 5), 0) # blur to make stars a bit bigger
        gray_blur = cv2.GaussianBlur(only_stars, (1, 1), 0)
        gray_blur = cv2.fastNlMeansDenoising(gray_blur, None, 1, 7, 21) # make them less nosy
        gray_blur = increase_brightness(gray_blur) # increase brightness to take all the stars into account
        
        gray_image = cv2.cvtColor(gray_blur, cv2.COLOR_BGR2GRAY) 

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)

        _, binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        cv2.imshow("binary", binary)
        cv2.waitKey(0)
        return image, binary
    

    def blob_detection1(self, scale = 3):
        # my first approach: scale image, determine centers of contours and map them on the original photo
        color_img, binary_img = self.adjust_image()

        original_height, original_width = binary_img.shape # store original dimensions
        
        # upscale images
        upscaled_color = cv2.resize(color_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        upscaled_binary = cv2.resize(binary_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


        edged = cv2.Canny(upscaled_binary, 0, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_centers = []

        for cnt in contours:
            cv2.drawContours(upscaled_color, [cnt], -1, (0, 255, 0), 1)

            x, y, w, h = cv2.boundingRect(cnt) # first option (star as rectangle)
            cX_rect = x + w // 2
            cY_rect = y + h // 2

            if len(cnt) >= 5: # second option (star as an ellipse, at least 5 points)
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (MA, ma), angle = ellipse  # center, major/minor axes, rotation
                cX_ellipse = int(cx)
                cY_ellipse = int(cy)
                cX, cY = cX_ellipse, cY_ellipse
            else:
                cX, cY = cX_rect, cY_rect

            cv2.circle(upscaled_color, (cX, cY), 2, (0, 0, 255), 3)

            # convert back to original scale
            scaled_cX = int(cX / scale)
            scaled_cY = int(cY / scale)
            shape_centers.append((scaled_cX, scaled_cY))


        # downscale for display
        transformed_color_down = cv2.resize(upscaled_color, 
                                            (original_width, original_height), 
                                            interpolation=cv2.INTER_AREA)

        cv2.imshow("Contours and centers", transformed_color_down)
        cv2.waitKey(0)

        return shape_centers

    def blob_detection2(self):
        # my second approach: used watershed algorithm to separate very close stars
        color_img, binary_img = self.adjust_image()
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8)) # create subplots with 2 rows and 2 columns
        
        # sure background area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) # define 1×1 rectangular structuring element
        sure_bg = cv2.dilate(binary_img, kernel, iterations=3) # expands the white regions
        axes[0, 0].imshow(sure_bg, cmap='gray')
        axes[0, 0].set_title('Sure Background') # everything that remains black after dilatation is background

        # distance transform
        dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5) # computes for each pixel in the foreground (white), the distance to the nearest background (black) pixel
        axes[0, 1].imshow(dist, cmap='gray')
        axes[0, 1].set_title('Distance Transform') # brighter values in dist mean the pixel is further inside a star and farther from the boundary

        # foreground area
        ret, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, cv2.THRESH_BINARY) # pick only those pixels that are definitely within a star
        sure_fg = sure_fg.astype(np.uint8)  
        axes[1, 0].imshow(sure_fg, cmap='gray')
        axes[1, 0].set_title('Sure Foreground')

        # unknown area
        unknown = cv2.subtract(sure_bg, sure_fg) # part of the sure background that is not in the sure foreground
        axes[1, 1].imshow(unknown, cmap='gray')
        axes[1, 1].set_title('Unknown')
        plt.show()

        ret, markers = cv2.connectedComponents(sure_fg) # markers is an image where each foreground star is assigned a label
        
        markers += 1 # add one to all labels so that background is not 0, but 1; 0 is for unknown reg
        markers[unknown == 255] = 0 # mark the region of unknown with zero
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(markers, cmap="tab20b")
        ax.axis('off')
        plt.show()

        # watershed algorithm: treats the input image like a topographical map
        markers = cv2.watershed(color_img, markers)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(markers, cmap="tab20b")
        ax.axis('off')
        plt.show()

        labels = np.unique(markers)
        stars = [] # collect star contour
        for label in labels[2:]: 
            # create a binary mask in which only the area of the label is in the foreground and the rest of the image is in the background 
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            # contour extraction on the created binary image
            contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            stars.append(contours[0])


        star_centers = []
        for marker in np.unique(markers):
            # skip background (e.g., marker==1) and border (often marker == -1)
            if marker <= 1:
                continue
            
            mask = np.uint8(markers == marker) # create a binary mask for the current label, star  reg
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                star_centers.append((cx, cy)) # get centers of stars
                cv2.circle(color_img, (cx, cy), 1, (0, 0, 255), -1)

        cv2.imshow("image", color_img)
        cv2.waitKey(0)
        return star_centers

if __name__ == "__main__":
    image_to_solve = "C:/Users/Andreea/Documents/constellation/dataset/test/images/2022-01-02-00-00-00-s_png_jpg.rf.da902dcc3763024472a80ca077612fcc.jpg"
    # image_to_solve = "C:/Users/Andreea/Documents/constellation/dataset/test/images/2022-01-09-00-00-00-s_png_jpg.rf.1b4788ef2a761e6133a58192102c6160.jpg"
    # image_to_solve = "C:/Users/Andreea/Documents/constellation/dataset/test/images/2022-01-11-00-00-00-s_png_jpg.rf.3b967c1738b7800202be12fc4fc19203.jpg"
    image_to_solve = "C:/Users/Andreea/Documents/constellation/dataset/test/images/2022-02-03-00-00-00-n_png_jpg.rf.249728aa17c0b712f336066d9595343b.jpg"
    # image_to_solve = "C:/Users/Andreea/Documents/constellation/2022-01-11-00-00-00-s_png_jpg.rf.3b967c1738b7800202be12fc4fc19203.jpg"
    detect = StarDetection(image_to_solve)
    stars = detect.blob_detection2()
   