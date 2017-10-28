import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from locate_pupil import locate_pupil

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--black', dest='consider_black', action='store_true')

    args = parser.parse_args()

    return args


def vis_detections(im, dets):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for det in dets:
        bbox = det[:4]
        score = det[-2]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(det[-1], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('eye and pupil detections'), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def main():
    args = parse_args()
    data_path = os.path.join('..', 'eye_test', 'data')
    image_set = os.path.join(data_path, 'ImageSets', 'test.txt')
    f = open(image_set, 'r')
    image_names = f.read().splitlines()
    f.close()

    for image_name in image_names:
        image_path = os.path.join(data_path, 'Images', image_name)
        im = cv2.imread(image_path)
        npy_path = os.path.join(data_path, 'Annotations', 
                    image_name.split('.')[0] + '_eye.npy')
        dets = np.load(npy_path)
        dets = np.uint16(np.around(dets))
        dets = np.ndarray.tolist(dets)
        
        pupils = []
        for det in dets:
            det.append('eye')
            eye = im[det[1] : det[3], det[0] : det[2]]
            pupil = locate_pupil(eye, det, args.consider_black)
            if pupil is not None:
                pupils.append(pupil)
        
        for pupil in pupils:
            dets.append(pupil)

        vis_detections(im, dets)
        plt.show()

if __name__ == '__main__':
    main()

