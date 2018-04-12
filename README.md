# Pupil-detection

### Dataset Structure
```
|__eye_test
   |__data
      |__Annotations
          |__*.npy
      |__Images
          |__*.jpeg
      |__ImageSets
          |__test.txt
```

### Visualize the pupil bounding boxes
```
cd ROOT_DIR/code
sh visualize.sh
```
### Generate xml files for pupil bounding boxes
```
cd ROOT_DIR/code
sh annotation.sh
```
