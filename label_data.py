

import cv2
import numpy as np
import os
import pickle

class KeypointsAnnotator:
    def __init__(self, num_keypoints=1):
        self.num_keypoints  = num_keypoints
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)

    def load_image(self, img):
        self.img = img
        self.vis = img.copy()
        #self.click_to_kpt = {0:"L", 1:"PULL", 2:"PIN", 3:"R"}

    def show_img(self, img):
        self.load_image(img)
        self.clicks = []
        self.label = 0

        cv2.imshow("pixel_selector", self.vis)

    def mouse_callback(self, event, x, y, flags, param):
        # if event == cv2.EVENT_LBUTTONDBLCLK:
        if event == cv2.EVENT_LBUTTONDOWN:
            #cv2.putText(img, self.click_to_kpt[len(self.clicks)], (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            self.clicks.append([x, y])
            cv2.circle(self.vis, (x, y), 3, (255, 0, 0), -1)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        self.label = 0

        cv2.imshow("pixel_selector", self.vis)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.clicks) == self.num_keypoints or (cv2.waitKey(33) == ord('s')):
                break
            if cv2.waitKey(33) == ord('r'):
                self.clicks = []
                self.load_image(img)
                print('Erased annotations for current image')
        print(self.clicks)
        return self.clicks

if __name__ == '__main__':
    pixel_selector = KeypointsAnnotator(num_keypoints=1)

    image_dir = '/Users/jennifergrannen/Documents/Stanford/iliad/vocal_sand/keypoints/data/4_26_pics_test' # Should have images like 00000.jpg, 00001.jpg, ...
    output_dir = '/Users/jennifergrannen/Documents/Stanford/iliad/vocal_sand/keypoints/data/gift_bag_test'
    annots_filename = 'annots.pkl' # Will have real_data/images and real_data/keypoints
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    full_annots = {} # {img_filename: {lang_ref: keypoint_annot}}

    if os.path.exists(os.path.join(output_dir, annots_filename)):
        with open(os.path.join(output_dir, annots_filename), 'rb') as f:
            full_annots = pickle.load(f)
    print("full_annots", [a.keys() for a in full_annots.values()][0])
    i = 0

    for f in sorted(os.listdir(image_dir)):
        if "jpg" in f:
            print("Img %d"%i)
            lang_ref = ""
            image_path = os.path.join(image_dir, f)
            print(image_path)
            img = cv2.imread(image_path)

            img = cv2.resize(img, (640, 480))

            img_filename = '%05d.jpg'%i
            image_outpath = os.path.join(output_dir, img_filename)

            lang_ref = "ball" # jellybean, lollipop

            # while not lang_ref == "done":
            #     # lang_ref = "pen"
            #     if lang_ref == "":
            #         pixel_selector.show_img(img)
            #
            #     lang_ref = input("Lang Ref?")
            #     if not lang_ref == "done":
            annots = pixel_selector.run(img)
            print("---")
            if len(annots)>0:
                annots = np.array(annots)
                cv2.imwrite(image_outpath, img)
                if img_filename not in full_annots.keys():
                    full_annots[img_filename] = {}
                full_annots[img_filename][lang_ref] = annots
                # np.save(keypoints_outpath, annots)
            i  += 1

    with open(os.path.join(output_dir, annots_filename), 'wb') as f:
        pickle.dump(full_annots, f)
