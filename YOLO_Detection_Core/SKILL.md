
#控制Con R G B三行的显示顺序
Order_Con_R_G_B = 'ConRGB' #'ConRGB,ConRBG,ConGRB,ConGBR,ConBRG,ConBGR' 多选一
#控制R G B行显示的颜色
color_R = (0, 0, 255) #可以更改
color_G = (0, 255, 0) #可以更改
color_B = (255, 0, 0) #可以更改
#控制No. 与cuvette之间的间距
distance_between_No_cuvette = 50 #可以更改
#控制cuvette和Con之间的间距
distance_between_cuvette_Con = 50 #可以更改
#控制Con、R、G、B之间的间距
distance_between_Con_R_G_B = 10 #可以更改

./interface.py 文件中新加了几个变量，分别是：
- Order_Con_R_G_B：控制Con R G B三行的显示顺序，可以选择 'ConRGB', 'ConRBG', 'ConGRB', 'ConGBR', 'ConBRG', 'ConBGR' 中的一个。
- color_R：控制R行显示的颜色，可以更改为其他RGB值。
- color_G：控制G行显示的颜色，可以更改为其他RGB值。
- color_B：控制B行显示的颜色，可以更改为其他RGB值。
- distance_between_No_cuvette：控制No.与cuvette之间的间距，可以更改为其他数值。
- distance_between_cuvette_Con：控制cuvette和Con之间的间距，可以更改为其他数值。
- distance_between_Con_R_G_B：控制Con、R、G、B之间的间距，可以更改为其他数值。
这些变量可以根据需要进行调整，以改变界面中各元素的显示顺序、颜色和间距。

./ultralytics/engine/results.py 文件中将导入./interface.py 文件中的新增的这些变量，他们分别是：Order_Con_R_G_B, color_R, color_G, color_B, distance_between_No_cuvette, distance_between_cuvette_Con, distance_between_Con_R_G_B。

修改./ultralytics/engine/results.py 文件中的相关代码，使其能够根据这些变量来调整识别照片中各元素的显示顺序、颜色和间距。具体来说，可以在识别图片时，根据Order_Con_R_G_B变量来确定Con R G B三行的显示顺序（'ConRGB', 'ConRBG', 'ConGRB', 'ConGBR', 'ConBRG', 'ConBGR'分别代表不同的显示顺序），并使用color_R、color_G、color_B变量来分别设置R、G、B行的颜色。同时，使用distance_between_No_cuvette、distance_between_cuvette_Con、distance_between_Con_R_G_B变量来调整No.与cuvette之间的间距、cuvette和Con之间的间距以及Con、R、G、B之间的间距。这样，界面中的元素将根据这些变量的设置进行调整，以满足用户的需求。
前提，保证不改变其他功能的前提下，进行上述修改。

#控制No Con R G B五行显示的颜色
color_No = (255, 255, 255) #可以更改
color_Con = (255, 255, 255) #可以更改
./interface.py 文件中新加了几个变量，分别是:
- color_No：控制No.行显示的颜色，可以更改为其他RGB值。
- color_Con：控制Con行显示的颜色，可以更改为其他RGB值。
在./ultralytics/engine/results.py 文件中将导入./interface.py 文件中的新增的color_No和color_Con变量，并修改相关代码，使其能够根据这些变量来调整No.行和Con行的显示颜色。具体来说，在识别图片时，使用color_No变量来设置No.行的颜色，使用color_Con变量来设置Con行的颜色。这样，界面中的No.行和Con行将根据这些变量的设置进行调整，以满足用户的需求。同时，保证不改变其他功能的前提下进行上述修改。


#显示字体大小和类型
output_font_size = 60 #可以更改
output_font = "Arial.ttf" #可以更改

在./ultralytics/engine/results.py 文件中将导入./interface.py 文件中的新增的output_font_size和output_font变量，并修改相关代码，使其能够根据这些变量来调整显示字体的大小和类型。
相关原代码中控制字体大小和类型的代码是下面这几行：
annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size=60,
            font="Arial.ttf",
            # pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            pil=True,
            example=names,
        )

./ultralytics/engine/results.py 文件中的第403和404行中的两行包含6个变量：x0_con, y0_con, x1_con, y1_con，w_con, h_con，这些变量分别代表Con.框的左上角坐标（x0_con, y0_con）、右下角坐标（x1_con, y1_con）以及宽度（w_con）和高度（h_con）。这些变量的值是根据原始框的坐标（x0, y0, x1, y1）以及预设的比例（x0_ratio, y0_ratio, x1_ratio, y1_ratio）计算得出的。具体来说，x0_con和y0_con是通过将原始框的左上角坐标与预设比例进行加权平均计算得出的，而x1_con和y1_con则是通过将原始框的右下角坐标与预设比例进行加权平均计算得出的。最后，w_con和h_con分别是通过计算Con.框的宽度和高度得出的。这些变量在后续的代码中将被用来绘制Con.框以及显示相关信息。
以下是第403和404行中相关代码片段：
x0_con, y0_con, x1_con, y1_con = int(x1*x0_ratio+x0*(1-x0_ratio)), int(y1*y0_ratio+y0*(1-y0_ratio)), int(x1*x1_ratio+x0*(1-x1_ratio)), int(y1*y1_ratio+y0*(1-y1_ratio))
w_con, h_con = (x1_con - x0_con), (y1_con - y0_con)

./ultralytics/engine/results.py 文件中
第615行：
linear_reg_con_rgb_file = os.path.join(results_dir, 'linear_con_rgb.xlsx')
和第649行：
detection_file = os.path.join(results_dir, (os.path.split(detection_dict['Name'])[-1].split('.')[0] + '.xlsx')) # use '+' instead of use joint, because joint add '\'
分别表示线性回归结果保存的文件路径和检测结果保存的文件路径。这些路径是通过将results_dir与相应的文件名进行拼接得到的。linear_reg_con_rgb_file表示线性回归结果保存的Excel文件路径，而detection_file表示检测结果保存的Excel文件路径，其中detection_dict['Name']是原始图像的文件名，通过os.path.split和字符串操作来生成对应的Excel文件名。这些文件路径将被用来保存线性回归结果和检测结果，以便后续查看和分析。
在生成以上线性回归结果和检测结果的Excel文件时，将x0_con, y0_con, x1_con, y1_con，w_con, h_con这六个变量添加到相应的Excel文件中，以便在查看和分析线性回归结果和检测结果时能够同时看到Con.框的坐标和尺寸信息。这些变量可以帮助用户更好地理解检测结果与Con.框之间的关系，以及评估检测的准确性和性能。

该程序中，已经将每个检测出来的识别区域中的Name	No.	Con.	Red	Green	Blue 保存起来了，在Blue列后面添加了六个新的列，分别是x0_con, y0_con, x1_con, y1_con，w_con, h_con，这些列分别保存了Con.框的左上角坐标（x0_con, y0_con）、右下角坐标（x1_con, y1_con）以及宽度（w_con）和高度（h_con）。这样，在查看和分析检测结果时，用户不仅可以看到每个识别区域的Name、No.、Con.、Red、Green、Blue等信息，还可以同时看到Con.框的坐标和尺寸信息，从而更全面地了解检测结果与Con.框之间的关系，以及评估检测的准确性和性能。要确保每个识别区域的Name	No.	Con.	Red	Green	Blue 和x0_con, y0_con, x1_con, y1_con，w_con, h_con结果一一对应。Blue列和x0_con列之间用一个空列分隔开，以便更清晰地区分原有的检测结果和新增的Con.框信息。

搜索一下程序，告诉我在进行线性公式计算的时候，我需要用到的数据是怎么来的，在哪些文件的哪行？高亮出来。不要修改代码

我现在需要在interface.py中增加一个行矩阵，用来控制获取线性公式时所用到的数据点（x,y), x在con_list里面选取，y在用来计算线性公式的R/G/B数值，x和y一定要严格对应。新增的控制矩阵，长度和con_list保持一致，元素仅为0,1；0代表不选取该位置对应的(x,y)作为线性公式计算的数据点，1代表选取该位置对应的(x,y)作为线性公式计算的数据点。