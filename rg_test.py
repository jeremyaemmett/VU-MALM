from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

test_image = 'C:/Users/Jeremy/Desktop/test_image.jpg'
img = cv2.imread(test_image)

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(test_image)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

print(detection_result)

parts = ['Nose', 'Right eye inner', 'Right eye center', 'Right eye outer', 'Left eye inner', 'Left eye center',
         'Left eye outer', 'Right ear', 'Left ear', 'Right corner mouth', 'Left corner mouth',
         'Right shoulder', 'Left shoulder', 'Right elbow', 'Left elbow', 'Right wrist', 'Left wrist',
         'Right pinky knuckle', 'Left pinky knuckle', 'Right index knuckle', 'Left index knuckle',
         'Right thumb knuckle', 'Left thumb knuckle', 'Right hip', 'Left hip', 'Right knee', 'Left knee',
         'Right ankle', 'Left ankle', 'Right heel', 'Left heel', 'Right foot', 'Left foot']

xs, ys, zs = [], [], []
test = detection_result.pose_landmarks
print(len(test[0]))
for i in range(0, len(test[0])):
  xs.append(test[0][i].x)
  ys.append(test[0][i].y)
  zs.append(test[0][i].z)

#fig = plt.figure(figsize=(15, 7))
#gs0 = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
#                      left=0.1, right=0.9, bottom=0.05, top=0.95,
#                      wspace=0.4, hspace=0.3)
#ax = fig.add_subplot(gs0[0, 0], projection = '3d')
#p3d1 = ax.scatter(xs, ys, zs)
#
#for i in range(0, len(parts)):
#    ax.text(xs[i], ys[i], zs[i], parts[i], horizontalalignment='center',
#              verticalalignment='bottom', fontsize=8.0, color='black',
#              bbox=dict(facecolor='white', edgecolor='none', boxstyle='round',
#                        alpha=0.0))

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
plt.imshow(visualized_mask, alpha = 0.2)

plt.show()