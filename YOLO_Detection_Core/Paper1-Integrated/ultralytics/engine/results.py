# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode


class BaseTensor(SimpleClass):
    """Base tensor class with additional methods for easy manipulation and device handling."""

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with data and original shape.

        Args:
            data (torch.Tensor | np.ndarray): Predictions, such as bboxes, masks and keypoints.
            orig_shape (tuple): Original shape of image.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes, optional): Object containing detection bounding boxes.
        masks (Masks, optional): Object containing detection masks.
        probs (Probs, optional): Object containing class probabilities for classification tasks.
        keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.

    Methods:
        update(boxes=None, masks=None, probs=None, obb=None): Updates object attributes with new detection results.
        cpu(): Returns a copy of the Results object with all tensors on CPU memory.
        numpy(): Returns a copy of the Results object with all tensors as numpy arrays.
        cuda(): Returns a copy of the Results object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the Results object with tensors on a specified device and dtype.
        new(): Returns a new Results object with the same image, path, and names.
        plot(...): Plots detection results on an input image, returning an annotated image.
        show(): Show annotated results to screen.
        save(filename): Save annotated results to file.
        verbose(): Returns a log string for each task, detailing detections and classifications.
        save_txt(txt_file, save_conf=False): Saves detection results to a text file.
        save_crop(save_dir, file_name=Path("im.jpg")): Saves cropped detection images.
        tojson(normalize=False): Converts detection results to JSON format.
    """

    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None) -> None:
        """
        Initialize the Results class.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.tensor, optional): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.tensor, optional): A 2D tensor of oriented bounding box coordinates for each detection.
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = {"preprocess": None, "inference": None, "postprocess": None}  # milliseconds per image
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        return self._apply("__getitem__", idx)

    def __len__(self):
        """Return the number of detections in the Results object."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            Results: A new Results object with attributes modified by the applied function.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all tensors on CPU memory."""
        return self._apply("cpu")

    def numpy(self):
        """Return a copy of the Results object with all tensors as numpy arrays."""
        return self._apply("numpy")

    def cuda(self):
        """Return a copy of the Results object with all tensors on GPU memory."""
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Return a copy of the Results object with tensors on the specified device and dtype."""
        return self._apply("to", *args, **kwargs)

    def new(self):
        """Return a new Results object with the same image, path, and names."""
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability
            show (bool): Whether to display the annotated image directly.
            save (bool): Whether to save the annotated image to `filename`.
            filename (str): Filename to save image to if save is True.

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('bus.jpg')  # results list
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.show()  # show image
                im.save('results.jpg')  # save image
            ```
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            print('my detect++++++------------------------------------')
            print('show_boxes=', show_boxes)
            from interface import (x0_ratio, y0_ratio, x1_ratio, y1_ratio, mode, con_list,
                                    rgb_calculate_accuracy,rgb_display_accuracy, con_display_accuracy,
                                   color_channel, results_dir, add_light)
            from scipy import stats
            import os, json

            # print('pred_boxes cls=', pred_boxes.cls)
            # print('pred_boxes xyxy=', pred_boxes.xyxy)
            # print('pred_boxes xywh=', pred_boxes.xywh)
            # print('pred_boxes type=', type(pred_boxes))
            # print('items =', pred_boxes.cls[0].item())

            index = 0
            # print('pred_boxes.cls', pred_boxes.cls)
            # b_avg_list.clear()

            # formula_path = os.path.join(os.path.dirname(os.path.dirname(self.path)), 'formula')
            # if not os.path.exists(formula_path):
            #     os.makedirs(formula_path)
            # # print('fromula_path:', formula_path)
            # formula_file = os.path.join(formula_path, 'formula.txt')
            # blue_file = os.path.join(formula_path, 'blue.txt')
            # # print('formula file name: ', formula_file)
            coor_list = []
            # store all the sample information in each image for both linear stage and detection stage
            linear_reg_dict = {'Name': str, 'No.': [], 'Con.': [], 'Red': [], 'Green': [], 'Blue': []}
            linear_reg_R_formula_dict = {'slope': [], 'intercept': [], 'r': [], 'R2': [], 'p': [], 'std_err': []}
            linear_reg_G_formula_dict = {'slope': [], 'intercept': [], 'r': [], 'R2': [], 'p': [], 'std_err': []}
            linear_reg_B_formula_dict = {'slope': [], 'intercept': [], 'r': [], 'R2': [], 'p': [], 'std_err': []}
            detection_dict = {'Name': str, 'No.': [], 'Con.': [], 'Red': [], 'Green': [], 'Blue': []}
            img_name = os.path.split(self.path)[-1].split('.')[0]
            linear_reg_dict['Name'] = img_name
            detection_dict['Name'] = img_name
            # print('img_name=', img_name)
            #check mode and color channel, the default is mode='linear', color_channel='B'
            if mode not in ['linear', 'detection']: mode = 'linear'
            if color_channel not in ['R', 'G', 'B']: color_channel = 'B'

            # sort the coordinates of the expected class
            for item in pred_boxes.cls:
                # print("type of item :", type(pred_boxes)
                if item.item() == 1:
                    # print('item=', item.names)
                    x0, y0, x1, y1 = pred_boxes.xyxy[index][0].item(), pred_boxes.xyxy[index][1].item(), pred_boxes.xyxy[index][2].item(), pred_boxes.xyxy[index][3].item()
                    _, _,  w, h = pred_boxes.xywh[index][0].item(), pred_boxes.xywh[index][1].item(), pred_boxes.xywh[index][2].item(), pred_boxes.xywh[index][3].item()
                    coor_list.append((x0, y0, x1, y1, w, h))
                    # print('coor_list:', coor_list)
                    # print('coor', coor)
                    # print('seperate:', coor[0], coor[1], coor[2], coor[3], coor[4], coor[5])
                index = index + 1
            coor_list.sort() # sort the coordinates by x0

            # calculate and annote the concentration and RGB
            id = 1 # the number of the cuvette, form left to right
            have_table = False
            if have_table: x0_last = 0
            if have_table: overall_list = [('No.', 'Con.', 'Blue', 'Green', 'Red')] # the overall list of the ids, concentrations, and RGBs
            id_dict = {} # save the ids and their corresponding positions in a dictionary
            light_dict = {} # save the lights and their corresponding positions in a dictionary
            # handle each sample in one image
            for coor in coor_list:
                # print('coor', coor)
                # print('seperate:', coor[0], coor[1], coor[2], coor[3], coor[4], coor[5])
                x0, y0, x1, y1, w, h = coor[0], coor[1], coor[2], coor[3], coor[4], coor[5]
                # print('xywh', x0, y0, x1, y1, w, h)

            # # for item in pred_boxes.cls:
            # #     # print("type of item :", type(pred_boxes))
            # #     # sort the coordinate of the expected class
            # #     if item.item() == 1:
            # #         # print('item=', item.names)
            # #         x0, y0, x1, y1 = pred_boxes.xyxy[index][0].item(), pred_boxes.xyxy[index][1].item(), pred_boxes.xyxy[index][2].item(), pred_boxes.xyxy[index][3].item()
            # #         _, _,  w,  h = pred_boxes.xywh[index][0].item(), pred_boxes.xywh[index][1].item(), pred_boxes.xywh[index][2].item(), pred_boxes.xywh[index][3].item()
            #
                y_bias = 110
                txt_bias = 70
                if False: #center point and extend area
                    x_ratio = 1/2
                    y_ratio = 2/3
                    xmid, ymid = int(x1*x_ratio+x0*(1-x_ratio)), int(y1*y_ratio+y0*(1-y_ratio))

                    x_pixbias = 5
                    y_pixbias = 5

                    x0_con, y0_con, w_con, h_con = xmid-x_pixbias, ymid-y_pixbias, x_pixbias*2+1, y_pixbias*2+1
                    x1_con, y1_con = x0_con+w_con, y0_con+h_con

                else: # start point and end point
                    # x0_ratio = 1/8
                    # y0_ratio = 1/2
                    # x1_ratio = 7/8
                    # y1_ratio = 7/8
                    x0_con, y0_con, x1_con, y1_con = int(x1*x0_ratio+x0*(1-x0_ratio)), int(y1*y0_ratio+y0*(1-y0_ratio)), int(x1*x1_ratio+x0*(1-x1_ratio)), int(y1*y1_ratio+y0*(1-y1_ratio))
                    w_con, h_con = (x1_con - x0_con), (y1_con - y0_con)
                    # print('xywh_con', x0_con, y0_con, x1_con, y1_con, w_con, h_con)
                # calculate the average RGB values of a sample in one image
                r_avg, g_avg, b_avg = self.calAvgRgb(annotator.im, x0_con, y0_con, w_con, h_con, rgb_calculate_accuracy)

                # mark the concentration area
                mybox = torch.tensor([x0_con, y0_con, x1_con, y1_con], device='cuda:0')
                annotator.box_label(mybox, label="", color=(250,240,10))

                if mode == 'linear':
                    # linear_reg_dict['Name']
                    # record the RGB and concentration in linear regression stage
                    linear_reg_dict['No.'].append(id)
                    # print('id-1', id-1)
                    linear_reg_dict['Con.'].append(con_list[id-1])
                    linear_reg_dict['Red'].append(r_avg)
                    linear_reg_dict['Green'].append(g_avg)
                    linear_reg_dict['Blue'].append(b_avg)

                    # b_avg_list.append(b_avg)
                    # c_con = con_list[id-1]

                if mode == 'detection':
                    # record the RGB in detection stage
                    detection_dict['No.'].append(id)
                    detection_dict['Red'].append(r_avg)
                    detection_dict['Green'].append(g_avg)
                    detection_dict['Blue'].append(b_avg)


                    # read the linear regression formula
                    # if os.path.exists(formula_file):
                    # pass read channel formula
                    if color_channel == 'R':
                        # linear_reg_R_formula_dict
                        data_dic = self.excelTodict(os.path.join(results_dir, 'linear_formula_R.xlsx'))
                    elif color_channel == 'G':
                        # linear_reg_G_formula_dict
                        data_dic = self.excelTodict(os.path.join(results_dir, 'linear_formula_G.xlsx'))
                    else: # the default is 'B'
                        # linear_reg_B_formula_dict
                        data_dic = self.excelTodict(os.path.join(results_dir, 'linear_formula_B.xlsx'))

                    # with open(formula_file, 'r') as file:
                    #     data_dic = json.load(file)
                    #     # print('data_dic:', data_dic)
                    #     # print('type(data_dic):', type(data_dic))
                    #     # print(data_dic['intercept'])
                    #     # print(data_dic['slope'])

                    # which channel to use
                    if color_channel == 'R':
                        which_channel = r_avg
                    elif color_channel == 'G':
                        which_channel = g_avg
                    else: # the default is 'B'
                        which_channel = b_avg
                    # calculate concentration according to linear equation
                    # print('type-data_dic[intercept]=', type(data_dic['intercept']))
                    c_con = (which_channel - data_dic['intercept'][0]) / data_dic['slope'][0] #data_dic['intercept'] is a list
                    # # record the concentration in detection stage
                    # print('type-c_con=', type(c_con))
                    detection_dict['Con.'].append(c_con)

                    # annotator.rectangle(xy=(x0_con, y0_con), width=5)
                    # annotator.box_label(box=Boxes(torch.Tensor([x0_con, y0_con, x1_con, y1_con,  0.9, 0]), orig_shape=annotator.im.shape))

                    # from PIL import ImageDraw
                    # self.im = im if input_is_pil else Image.fromarray(im)
                    # print('img =', img)
                    # self.draw = ImageDraw.Draw(img)
                    # self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)

                    # box1 = Boxes(boxes=None, self.orig_shape) #if boxes is not None else None
                    # box1 = Boxes(None, self.orig_shape) #if boxes is not None else None
                    # annotator.box_label(box=box1, label='YUe Hengmao')

                # truncate RGB and concentration for display
                if mode == 'linear':
                    # print('ckkkkkkkk=', round(55.5535, rgb_display_accuracy))
                    # print('linear_reg_dict[\'Red\'][id - 1]=', linear_reg_dict['Red'][id - 1])
                    # print('type - linear_reg_dict[\'Red\'][id - 1]=', type(linear_reg_dict['Red'][id - 1]))
                    con_dis = round(linear_reg_dict['Con.'][id - 1], con_display_accuracy)
                    r_dis = round(linear_reg_dict['Red'][id - 1], rgb_display_accuracy)
                    g_dis = round(linear_reg_dict['Green'][id - 1], rgb_display_accuracy)
                    b_dis = round(linear_reg_dict['Blue'][id - 1], rgb_display_accuracy)
                else: # default is detection
                    # print('ckkkkkkkk=', round(56.7535, con_display_accuracy))
                    # print('detection_dict[\'Con.\'][id - 1]=', detection_dict['Con.'][id - 1])
                    # print('type - detection_dict[\'Con.\'][id - 1]=', type(detection_dict['Con.'][id - 1]))
                    # print('weweewekkkk=', round(89.7535, rgb_display_accuracy))
                    # print('detection_dict[\'Red\'][id - 1]=', detection_dict['Red'][id - 1])
                    # print('type - detection_dict[\'Red\'][id - 1]=', type(detection_dict['Red'][id - 1]))
                    con_dis = round(detection_dict['Con.'][id - 1], con_display_accuracy)
                    r_dis = round(detection_dict['Red'][id - 1], rgb_display_accuracy)
                    g_dis = round(detection_dict['Green'][id - 1], rgb_display_accuracy)
                    b_dis = round(detection_dict['Blue'][id - 1], rgb_display_accuracy)


                # print('b_avg = ', b_avg)
                # print('g_avg = ', g_avg)
                # print('green_text = ', green_text)

                annotator.text([int(x0), int(y1) + y_bias], "Con.:" + str(con_dis), txt_color=(0, 0, 0))
                annotator.text([int(x0), int(y1) + y_bias + txt_bias * 1], "G:" + str(g_dis), txt_color=(0, 255, 0))
                annotator.text([int(x0), int(y1) + y_bias + txt_bias * 2], "B:" + str(b_dis), txt_color=(255, 0, 0))
                annotator.text([int(x0), int(y1) + y_bias + txt_bias * 3], "R:" + str(r_dis), txt_color=(0, 0, 255))

                # annotator.text([int(x0), int(y1) - y_bias * 8 - txt_bias * 2], "No." + str(id), txt_color=(0, 0, 0))
                if add_light:
                    if con_dis < 5:
                        light = os.path.join(os.getcwd(), 'custom/lightImg/green_BGR.png')
                    elif con_dis > 30:
                        light = os.path.join(os.getcwd(), 'custom/lightImg/red_BGR.png')
                    else: # 5<con_dis<30
                        light = os.path.join(os.getcwd(), 'custom/lightImg/yellow_BGR.png')
                    light_dict[str(id)] = [int(x0)+15, int(y1) - y_bias * 8 - txt_bias * 2 - 300, light]
                id_dict[str(id)] = [int(x0)+50, int(y1) - y_bias * 8 - txt_bias * 2]

                # add c_con, b_avg, g_avg, r_avg to the overall list
                if have_table: overall_list.append((id, c_con, b_avg, g_avg, r_avg))
                # the x0 of the last sample
                if have_table: x0_last = x0
                # print('x0_last =================== x0:', x0)
                id = id + 1
            # draw the id and show the lights
            # make the id have the same y position
            y_min = min([id_dict[key][1] for key in id_dict.keys()])
            for key in id_dict.keys():
                id_dict[key][1] = y_min
            # annotate the image with ids
            for key in id_dict.keys():
                annotator.text(id_dict[key], "No." + key, txt_color=(0, 0, 0))
                # print(id_dict[key])
                # print(min([id_dict[key][1] for key in id_dict.keys()]))
            # light_dict
            if add_light:
                # make the light position have the same y position
                ylight_min = min([light_dict[key][1] for key in light_dict.keys()])
                for key in light_dict.keys():
                    light_dict[key][1] = ylight_min
                annotator.pasteimage(light_size=200, light_dict=light_dict)

            # print('img_name=', img_name)
            # print('self.path=', self.path)
            # print('type(im)=', type(self.orig_img))
            # print('self.orig_img=', self.orig_img)
            # print('annotator.im:', annotator.im)
            # print('annotator.im-type:', type(annotator.im))


            # index = index + 1 # archieve
            # coor_list.sort() # archieve
            # id = 1
            # for item in coor_list:
            #     annotator.text([int(item[0]), int(item[1]) - 630], "ID=" + str(id), txt_color=(255, 255, 255))
            #     id = id + 1
            if have_table:
                # x0_last = 0
                x0_last_bias = 300
                y_center = 1512 + 250
                h_line = 100
                num_lines = len(coor_list) + 1
                x0_overall = x0_last + x0_last_bias
                # print('x_last:', x0_last, 'x_overall:', x0_overall)
                y0_overall = y_center - (num_lines * h_line) / 2
                overall_index = 0
                annotator.text([int(x0_overall-50), int(y0_overall - h_line * 1.5)], 'TABLE. Concentration and RGB', txt_color=(255, 255, 255))
                for overall_item in overall_list:
                    annotator.text([int(x0_overall), int(y0_overall + h_line * overall_index)], str(overall_item[0]), txt_color=(255, 255, 255))
                    annotator.text([int(x0_overall + 150), int(y0_overall + h_line * overall_index)], str(overall_item[1]), txt_color=(255, 255, 255))
                    annotator.text([int(x0_overall + 300), int(y0_overall + h_line * overall_index)], str(overall_item[2]), txt_color=(255, 145, 48))
                    annotator.text([int(x0_overall + 450), int(y0_overall + h_line * overall_index)], str(overall_item[3]), txt_color=(0, 255, 0))
                    annotator.text([int(x0_overall + 630), int(y0_overall + h_line * overall_index)], str(overall_item[4]), txt_color=(0, 0, 255))
                    overall_index = overall_index + 1

            if mode == 'linear':
                # check if the number of concentration equal to RGB value
                if len(linear_reg_dict['Con.']) == len(linear_reg_dict['Red']):
                    # files to save concentration, rgb, formula R,G,B ,seperately in the linear stage
                    linear_reg_con_rgb_file = os.path.join(results_dir, 'linear_con_rgb.xlsx')
                    linear_reg_R_formula_file = os.path.join(results_dir, 'linear_formula_R.xlsx')
                    linear_reg_G_formula_file = os.path.join(results_dir, 'linear_formula_G.xlsx')
                    linear_reg_B_formula_file = os.path.join(results_dir, 'linear_formula_B.xlsx')
                    text = ['slope', 'intercept', 'r', 'R2', 'p', 'std_err']
                    # save the linear regression formula of Red vs. Con. to the file ...linear_formula_R.xlsx
                    slope, intercept, r, p, std_err = stats.linregress(linear_reg_dict['Con.'], linear_reg_dict['Red'])
                    paras = [slope, intercept, r, r * r, p, std_err]
                    for _, value in enumerate(zip(text, paras)): linear_reg_R_formula_dict[value[0]].append(value[1])
                    # print('linear_reg_R_formula_dict', linear_reg_R_formula_dict)
                    self.dictToexcel(linear_reg_R_formula_dict, linear_reg_R_formula_file)
                    # save the linear regression formula of Green vs. Con. to the file ...linear_formula_G.xlsx
                    slope, intercept, r, p, std_err = stats.linregress(linear_reg_dict['Con.'], linear_reg_dict['Green'])
                    paras = [slope, intercept, r, r * r, p, std_err]
                    for _, value in enumerate(zip(text, paras)): linear_reg_G_formula_dict[value[0]].append(value[1])
                    # print('linear_reg_G_formula_dict', linear_reg_G_formula_dict)
                    self.dictToexcel(linear_reg_G_formula_dict, linear_reg_G_formula_file)
                    # save the linear regression formula of Blue vs. Con. to the file ...linear_formula_B.xlsx
                    slope, intercept, r, p, std_err = stats.linregress(linear_reg_dict['Con.'], linear_reg_dict['Blue'])
                    paras = [slope, intercept, r, r * r, p, std_err]
                    for _, value in enumerate(zip(text, paras)): linear_reg_B_formula_dict[value[0]].append(value[1])
                    # print('linear_reg_B_formula_dict', linear_reg_B_formula_dict)
                    self.dictToexcel(linear_reg_B_formula_dict, linear_reg_B_formula_file)
                    # save RGB values of linear image into file ...linear_con_rgb
                    self.dictToexcel(linear_reg_dict, linear_reg_con_rgb_file)

                    # # save to excel file. input: filename, p
                    # with open(formula_file, 'w') as file:
                    #     json.dump({'slope':slope, 'intercept':intercept, 'r':r}, file)
                    # with open(blue_file, 'w') as file:
                    #     json.dump(b_avg_list, file)
                    # # os.path.dirname(self.path)
            else: # detection mode
                # print(os.path.split(detection_dict['Name'])[-1].split('.')[0])
                detection_file = os.path.join(results_dir, (os.path.split(detection_dict['Name'])[-1].split('.')[0] + '.xlsx')) # use '+' instead of use joint, because joint add '\'
                # print('detection_file', detection_file)
                self.dictToexcel(detection_dict, detection_file)

            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name}{conf:.2f}" if conf else name) if labels else None
                # label = f"{name}" if name else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                # print('box=', box)
                annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)
                # print('color=', colors(c, True))


        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            # print('save.filename=', filename)
            annotator.save(filename)
        # print('status(save)=', save)
        return annotator.result()


# ============================
    def dictToexcel(self, input_dict, file_path):
        """save dictionary to an excel"""
        df = pd.DataFrame.from_dict(input_dict)
        df.to_excel(file_path, index=False)

    def excelTodict(self, file_path):
        """read data from dictionary to save as a dictionary"""
        df = pd.read_excel(file_path)
        dict_df = df.to_dict()
        # print(type(dict_df))
        # print(dict_df)
        dict_result = {}
        for key in dict_df.keys():
            dict_result[key] = list(dict_df[key].values())
            # print(list(dict_df[key].values()))
        # print(dict_result)
        return dict_result  # {key:list[]}

    def calAvgRgb(self, img, x, y, w, h, accuracy=16):
        """calculate the average RGB values of a selected area to a certain accuracy"""
        if 0 == w or 0 == h:
            return False
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for width in range(w):
            for height in range(h):
                b, g, r = img[y + height, x + width]
                #                print('h:',h,'s:',s,'v:',v)
                r_sum = r_sum + r
                g_sum = g_sum + g
                b_sum = b_sum + b
        #        print('a_sum =', a_sum)
        #        print('b_sum =', b_sum)
        #        print('c_sum =', c_sum)

        self.accuracy = accuracy
        # print('w*h=', w*h)
        #        print('self.accuracy', self.accuracy)
        if self.accuracy == 0:
            r_avg = round(r_sum / (w * h))
            g_avg = round(g_sum / (w * h))
            b_avg = round(b_sum / (w * h))
        else:
            r_avg = round(r_sum / (w * h), self.accuracy)
            g_avg = round(g_sum / (w * h), self.accuracy)
            b_avg = round(b_sum / (w * h), self.accuracy)
        #        print('a_avg', a_avg, 'b_avg', b_avg, 'c_avg', c_avg)

        return r_avg, g_avg, b_avg

# ============================


    def show(self, *args, **kwargs):
        """Show annotated results image."""
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """Save annotated results image."""
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        print('filenameinsave=', filename)
        return filename

    def verbose(self):
        """Return log string for each task."""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Save cropped predictions to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            file_name (str | pathlib.Path): File name.
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """Convert the results to a summarized format."""
        if self.probs is not None:
            LOGGER.warning("Warning: Classify results do not support the `summary()` method yet.")
            return

        # Create list of detection dictionaries
        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            box = {
                "x1": round(row[0] / w, decimals),
                "y1": round(row[1] / h, decimals),
                "x2": round(row[2] / w, decimals),
                "y2": round(row[3] / h, decimals),
            }
            conf = round(row[-2], decimals)
            class_id = int(row[-1])
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": box}
            if self.boxes.is_track:
                result["track_id"] = int(row[-3])  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def tojson(self, normalize=False, decimals=5):
        """Convert the results to JSON format."""
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)


class Boxes(BaseTensor):
    """
    Manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class
    identifiers, and optional tracking IDs. Supports multiple formats for box coordinates, including both absolute and
    normalized forms.

    Attributes:
        data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.

    Properties:
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray, optional): Tracking IDs for each box, if available.
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, calculated on demand.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes, relative to `orig_shape`.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes, relative to `orig_shape`.

    Methods:
        cpu(): Moves the boxes to CPU memory.
        numpy(): Converts the boxes to a numpy array format.
        cuda(): Moves the boxes to CUDA (GPU) memory.
        to(device, dtype=None): Moves the boxes to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class.

        Args:
            boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes, with
                shape (num_boxes, 6) or (num_boxes, 7). The last two columns contain confidence and class values.
                If present, the third last column contains track IDs.
            orig_shape (tuple): Original image size, in the format (height, width).
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    Attributes:
        xy (list): A list of segments in pixel coordinates.
        xyn (list): A list of normalized segments.

    Methods:
        cpu(): Returns the masks tensor on CPU memory.
        numpy(): Returns the masks tensor as a numpy array.
        cuda(): Returns the masks tensor on GPU memory.
        to(device, dtype): Returns the masks tensor with the specified device and dtype.
    """

    def __init__(self, masks, orig_shape) -> None:
        """Initialize the Masks class with the given masks tensor and original image shape."""
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Return normalized segments."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Return segments in pixel coordinates."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    Attributes:
        xy (torch.Tensor): A collection of keypoints containing x, y coordinates for each detection.
        xyn (torch.Tensor): A normalized version of xy with coordinates in the range [0, 1].
        conf (torch.Tensor): Confidence values associated with keypoints if available, otherwise None.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(device, dtype): Returns a copy of the keypoints tensor with the specified device and dtype.
    """

    @smart_inference_mode()  # avoid keypoints < conf in-place error
    def __init__(self, keypoints, orig_shape) -> None:
        """Initializes the Keypoints object with detection keypoints and original image size."""
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # points with conf < 0.5 (not visible)
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Returns x, y coordinates of keypoints."""
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Returns normalized x, y coordinates of keypoints."""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """Returns confidence values of keypoints if available, else None."""
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification predictions.

    Attributes:
        top1 (int): Index of the top 1 class.
        top5 (list[int]): Indices of the top 5 classes.
        top1conf (torch.Tensor): Confidence of the top 1 class.
        top5conf (torch.Tensor): Confidences of the top 5 classes.

    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(): Returns a copy of the probs tensor with the specified device and dtype.
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """Initialize the Probs class with classification probabilities and optional original shape of the image."""
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """Return the index of top 1."""
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """Return the indices of top 5."""
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """Return the confidence of top 1."""
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """Return the confidences of top 5."""
        return self.data[self.top5]


class OBB(BaseTensor):
    """
    A class for storing and manipulating Oriented Bounding Boxes (OBB).

    Args:
        boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values.
            If present, the third last column contains track IDs, and the fifth column from the left contains rotation.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        xywhr (torch.Tensor | numpy.ndarray): The boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray): The track IDs of the boxes (if available).
        xyxyxyxyn (torch.Tensor | numpy.ndarray): The rotated boxes in xyxyxyxy format normalized by orig image size.
        xyxyxyxy (torch.Tensor | numpy.ndarray): The rotated boxes in xyxyxyxy format.
        xyxy (torch.Tensor | numpy.ndarray): The horizontal boxes in xyxyxyxy format.
        data (torch.Tensor): The raw OBB tensor (alias for `boxes`).

    Methods:
        cpu(): Move the object to CPU memory.
        numpy(): Convert the object to a numpy array.
        cuda(): Move the object to CUDA memory.
        to(*args, **kwargs): Move the object to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """Initialize the Boxes class."""
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        """Return the rotated boxes in xywhr format."""
        return self.data[:, :5]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        """Return the boxes in xyxyxyxy format, (N, 4, 2)."""
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        """Return the boxes in xyxyxyxy format, (N, 4, 2)."""
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        """
        Return the horizontal boxes in xyxy format, (N, 4).

        Accepts both torch and numpy boxes.
        """
        x1 = self.xyxyxyxy[..., 0].min(1).values
        x2 = self.xyxyxyxy[..., 0].max(1).values
        y1 = self.xyxyxyxy[..., 1].min(1).values
        y2 = self.xyxyxyxy[..., 1].max(1).values
        xyxy = [x1, y1, x2, y2]
        return np.stack(xyxy, axis=-1) if isinstance(self.data, np.ndarray) else torch.stack(xyxy, dim=-1)
