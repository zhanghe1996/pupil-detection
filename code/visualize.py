import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET

from locate_pupil import locate_pupil

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--visualize', dest='if_visualize', action='store_true')
    parser.add_argument('--annotation', dest='generate_annotation', action='store_true')

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
    output_path = os.path.join('..', 'Annotations')
    f = open(image_set, 'r')
    image_names = f.read().splitlines()
    f.close()

    for image_name in image_names:
        image_path = os.path.join(data_path, 'Images', image_name + '.jpeg')

        if not os.path.isfile(image_path):
            continue
        
        # load image
        im = cv2.imread(image_path)
        # load eye positions from npy file
        npy_path = os.path.join(data_path, 'Annotations', 
                    image_name.split('.')[0] + '_eye.npy')
        dets = np.load(npy_path)
        dets = np.uint16(np.around(dets))
        dets = np.ndarray.tolist(dets)

        remove = []
        overlap = [False] * len(dets)
        # if there are more than two eyes detected, remove all overlapped smaller regions.
        if len(dets) > 2:
            for i in range(0, len(dets)):
                for j in range(i + 1, len(dets)):
                    if dets[i][1] <= dets[j][3] and dets[i][3] >= dets[j][1]:
                        overlap[i] = True
                        overlap[j] = True
            
            for i in range(0, len(dets)):
                if not overlap[i]:
                    remove.append(i)

            dets = np.ndarray.tolist(np.delete(dets, remove, 0))

        pupils = []
        for det in dets:
            det.append('eye')
            eye = im[det[1] : det[3], det[0] : det[2]]
            # get pupil region within the eye region
            pupil = locate_pupil(eye, det, image_name.split('_')[1])
            if pupil is not None:
                pupils.append(pupil)
        
        for pupil in pupils:
            dets.append(pupil)

        # show the image with the bounding boxes of eye and pupil
        if args.if_visualize:
            vis_detections(im, dets)
            plt.show()

        # generate xml files in the Annotation folder
        if args.generate_annotation:
            annotation = ET.Element('annotation')
            for pupil in pupils:
                pupil_object = ET.SubElement(annotation, "object")
                bndbox = ET.SubElement(pupil_object, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(pupil[0])
                ET.SubElement(bndbox, "ymin").text = str(pupil[1])
                ET.SubElement(bndbox, "xmax").text = str(pupil[2])
                ET.SubElement(bndbox, "ymax").text = str(pupil[3])
            tree = ET.ElementTree(annotation)
            tree.write(os.path.join(output_path, image_name.split('.')[0] + '.xml'))



if __name__ == '__main__':
    main()

