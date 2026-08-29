from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('weights/yolo/yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('weights/yolo/yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='C:/Users/李肖夏/Desktop/Peng/Peng/custom/detectImg/cuvette_Peng.v1i.yolov8/data.yaml', epochs=300, device=0)
    metrics = model.val()  # 在验证集上评估模型性能

    # # Load a model
    # model = YOLO('yolov8s.yaml')  # build a new model from YAML
    # model = YOLO('weights/yolo/yolov8s.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8s.yaml').load('weights/yolo/yolov8s.pt')  # build from YAML and transfer weights
    #
    # # Train the model
    # results = model.train(data='C:/yue/eclipse/MLchem/ImgRec/YOLOv8/custom/mlsensing.v3-mlsensing_640.yolov8/data.yaml',
    #                       epochs=300, device=0)
    # metrics = model.val()  # 在验证集上评估模型性能


    # # Load a model
    # model = YOLO('yolov8m.yaml')  # build a new model from YAML
    # model = YOLO('weights/yolo/yolov8m.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8m.yaml').load('weights/yolo/yolov8m.pt')  # build from YAML and transfer weights
    #
    # # Train the model
    # results = model.train(data='C:/yue/eclipse/MLchem/ImgRec/YOLOv8/custom/mlsensing.v3-mlsensing_640.yolov8/data.yaml',
    #                       epochs=300, device=0)
    # metrics = model.val()  # 在验证集上评估模型性能


    # # Load a model
    # model = YOLO('yolov8l.yaml')  # build a new model from YAML
    # model = YOLO('weights/yolo/yolov8l.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8l.yaml').load('weights/yolo/yolov8l.pt')  # build from YAML and transfer weights
    #
    # # Train the model
    # results = model.train(data='C:/yue/eclipse/MLchem/ImgRec/YOLOv8/custom/mlsensing.v3-mlsensing_640.yolov8/data.yaml',
    #                       epochs=300, device=0)
    # metrics = model.val()  # 在验证集上评估模型性能

    # # Load a model
    # model = YOLO('yolov8x.yaml')  # build a new model from YAML
    # model = YOLO('weights/yolo/yolov8x.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8x.yaml').load('weights/yolo/yolov8x.pt')  # build from YAML and transfer weights
    #
    # # Train the model
    # results = model.train(data='C:/yue/eclipse/MLchem/ImgRec/YOLOv8/custom/mlsensing.v3-mlsensing_640.yolov8/data.yaml',
    #                       epochs=300, device=0)
    # metrics = model.val()  # 在验证集上评估模型性能






    #
    # from ultralytics import YOLO
    #
    # # 加载模型
    # model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    # model = YOLO("weights/yolov8n.pt")  # 加载预训练模型（建议用于训练）
    #
    # # 使用模型
    # model.train(data="coco128.yaml", epochs=3)  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能
    # results = model.predict(source="ultralytics/assets/bus.jpg", save=True)  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    # Terminal:

    # activate yolov8
    # pip install ultralytics