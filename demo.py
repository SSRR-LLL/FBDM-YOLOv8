import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("ultralytics/cfg/models/v8/yolov8s-my.yaml")  # build a new model from scratch
   # model = YOLO("D://yolodaima//ultralytics-main//yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data=r"D:\github\ultralytics-main\Target.yaml", epochs=300)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format
