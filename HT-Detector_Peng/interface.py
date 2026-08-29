
# For 小彭

import os
linear_dir = os.path.join(os.getcwd(), 'custom/linear_detection/linear/') #不用更改
detection_dir = os.path.join(os.getcwd(), 'custom/linear_detection/detection/') #不用更改
results_dir = os.path.join(os.getcwd(), 'runs/detect/results/') #不用更改
detect_weights_path = "weights/cuvette_Peng/yolov8n_train/weights/best.pt" #不用更改

#选择框的左上角点位置
x0_ratio = 3.4/8 #可以更改
y0_ratio = 4.3/8 #可以更改
#选择框的右下角点位置
x1_ratio = 4.6/8 #可以更改
y1_ratio = 5.7/8 #可以更改
#获取线性公式设置为 linear，获取识别结果设置为 detection
mode = 'detection' # 'linear' or 'detection' 可以更改只能选择这两个之一
#获取线性公式时，需要手动输入浓度值，个数一定严格和比色皿数量比配
con_list = [0, 0.01, 0.05, 0.1, 0.5, 1 ,2, 5, 10] #可以更改
#获取线性公式时，控制每个(con, RGB)数据点是否参与线性回归；长度必须和con_list一致，0不选，1选
linear_formula_point_matrix = [1, 1, 1, 1, 1, 1, 1, 0, 0] #可以更改
#计算RGB平均值用到的小数点位数，如不提供程序默认16位
rgb_calculate_accuracy = 16 #可以更改
#图片显示RGB和Con.的小数点位数，可以单独控制
rgb_display_accuracy = 2 #可以更改
con_display_accuracy = 2 #可以更改
#置信度阈值，只有置信度>=该值才判定为目标物体
detect_confidence = 0.05 #可以更改，范围0~1，值越大越严格
#是否在标签上显示置信度数值
show_confidence = False #True显示，False不显示
#选择相应的颜色通道
color_channel = 'B' # 'R' 'G' 'B' 三选一
#控制Con R G B三行的显示顺序
Order_Con_R_G_B = 'ConGBR' #'ConRGB,ConRBG,ConGRB,ConGBR,ConBRG,ConBGR' 多选一
#控制No Con R G B五行显示的颜色,格式为(B,G,R)，数值从0-255； (0, 0, 0)为黑色 (255, 255, 255)为白色
color_No = (255, 0, 255) #可以更改 
color_Con = (255, 255, 0) #可以更改 
color_R = (0, 0, 255) #可以更改
color_G = (0, 255, 0) #可以更改
color_B = (255, 0, 0) #可以更改
#控制No. 与cuvette之间的间距
distance_between_No_cuvette = 0 #可以更改
#控制cuvette和Con之间的间距
distance_between_cuvette_Con = 0 #可以更改
#控制Con、R、G、B之间的间距
distance_between_Con_R_G_B = 10 #可以更改
#显示字体大小和类型
output_font_size = 25 #可以更改
output_font = "Arial.ttf" #可以更改，谨慎更改 # "Arial.ttf"
#是否添加指示灯
add_light = False


# 升级功能
# 线性回归和浓度识别，照片分开文件夹放，linear_dir 和 detection_dir
# linear和detection结果都放在results_dir中
# 一定先linear再detection，linear会删除results_dir中所有excel结果
# linear产生的结果文件：linear_con_rgb.xlsx, linear_formula_R.xlsx，linear_formula_B.xlsx，linear_formula_G.xlsx
# detection产生的结果文件：xxx.xlsx
# excel中保存高精度结果。平均rgb计算小数点可控，显示输出结果可控。











