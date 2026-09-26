from ultralytics import YOLO
import os
import time

def detection(weight_path, img_dir):
    #
    model = YOLO(weight_path)  # 加载预训练模型（建议用于训练）
    file_names = os.listdir(img_dir)
    abs_file_names = []
    for file_name in file_names:
        abs_file_names.append(os.path.join(img_dir, file_name))
    for abs_file_name in abs_file_names:
        start = time.time()
        results = model.predict(source=abs_file_name, save=True)  # 对图像进行预测
        end = time.time()
        print('The whole time is :', (end-start)*1000, ' ms')




if __name__ == '__main__':
    # # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('weights/yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('weights/yolov8n.pt')  # build from YAML and transfer weights
    #
    # # Train the model
    # results = model.train(data='C:/yue/eclipse/MLchem/ImgRec/YOLOv8/custom/data.yaml', epochs=300, device=0)
    # metrics = model.val()  # 在验证集上评估模型性能




    # # 加载模型
    # model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    # model = YOLO("weights/mlsensing/yolov8n_640_best.pt")  # 加载预训练模型（建议用于训练）
    #
    # # 使用模型
    # model.train(data="coco128.yaml", epochs=3)  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能
    # results = model.predict(source="custom/detectImg/Date0420_S01ml_Num08_Deg000_Dist020cm.jpg", save=True)  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式




    #
    model = YOLO("weights/cuvette_huili/yolov8n_train/weights/best.pt")  # 加载预训练模型（建议用于训练）
    file_names = os.listdir(os.path.join(os.getcwd(), "custom/detectImg/5.15/"))
    abs_file_names = []
    for file_name in file_names:
        abs_file_names.append(os.path.join(os.getcwd(), "custom/detectImg/5.15/", file_name))
    # print('name:', abs_file_names, 'length:', len(abs_file_names))
    for abs_file_name in abs_file_names:
        print('abs_file_name:', abs_file_name)
        print('THE FIRST  TIME')
        results = model.predict(source=abs_file_name, save=True)  # 对图像进行预测

        # print('type(results)========')
        # print(type(results))
        # print('results-------')
        # print(results)
        # print('++++++++++++++++++++++++++++++++++++++++')
        # print(abs_file_name)

    # model = YOLO("weights/mlsensing/yolov8n_640_best.pt")  # 加载预训练模型（建议用于训练）
    # results = model.predict(source="custom/detectImg/Date0420_S01ml_Num03_Deg000_Dist020cm.jpg", save=True)  # 对图像进行预测
    #
    # # for result in results:
    # # print("++++++++++++++++++++++++++++++++++++++++++")
    # # print('type(result)', type(results))
    # # print(results.boxes.xywh)
    # # print("__________________________________________")
    #
    #
    # for result in results:
    #     print("++++++++++++++++++++++++++++++++++++++++++")
    #     print('len(results)', len(results))
    #     print('type(result)', type(result))
    #     print(result.boxes.xyxy)
    #     print("__________________________________________")
    #
    #
    #
    # # for item in results:
    # #     print('len(item)=', len(item))
    # #     print('type(item)=', type(item))
    # #     for iii in item:
    # #         print('len(iii)',len(iii))
    # #         print('type(iii)=','\n', type(iii))
    # #         print(iii, '\n','________________________________')
    # #     # print('item---------------------',"\n", item)

