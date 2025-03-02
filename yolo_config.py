########################## Updatable Configurations ##########################
# YOLO model version to use (options: yolov8s, yolov8m, yolov8l, yolov8x)
# Could try to use the latest version of YOLO models
yolo_model_version = 'yolo11m-seg'

# Whether to start fine-tuning from original weights each time when you trigger the training at label studio
fine_tune_start_from_original = False

# Model prediction configuration
# Confidence threshold and IoU threshold for predictions
# Additional parameters can be passed according to official Ultralytics documentation
# model_predict_config = dict(
#     conf=0.3,
#     iou=0.1
# )
model_predict_config = {}

# Model training configuration
# Parameters for training process, refer to official documentation for details
model_train_config = dict(
    data=r'datasets/data.yaml',
    epochs=1000,
    patience=20,
    plots=True,
    exist_ok=True,
    verbose=True
)
