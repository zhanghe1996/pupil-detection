import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from locate_pupil import locate_pupil

CONF_THRESH = 0.80

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--class', dest='my_class', help='Network to use [vgg16]',
                        default='__background__', type=str)

    args = parser.parse_args()

    return args


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def main():
    args = parse_args()
    image_set = os.path.join(args.my_class + '_test', 'data', 'ImageSets', 'test.txt')
    f = open(image_set, 'r')
    image_names = f.read().splitlines()
    f.close()

    for image_name in image_names:
        image_path = os.path.join(args.my_class + '_test', 'data', 'Images', image_name)
        im = cv2.imread(image_path)
        npy_path = os.path.join(args.my_class + '_test', 'data', 'Annotations', 
                    image_name.split('.')[0] + '_' + args.my_class + '.npy')
        det = np.load(npy_path)
        print det

        vis_detections(im, args.my_class, det, CONF_THRESH)

    plt.show()

if __name__ == '__main__':
    main()

