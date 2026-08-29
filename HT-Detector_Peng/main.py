import os, shutil
import detect
from interface import linear_dir, detection_dir, results_dir, detect_weights_path, mode

# only delete the results directory at the linear regression stage
if mode == 'linear':
    # if path not exists, create
    if not os.path.exists(results_dir):
        os.makedirs(results_dir) # 父目录缺失会补上，优先创建目录 https://www.geeksforgeeks.org/create-a-directory-in-python/#makedirs
    # if dir not empty, clean
    if os.listdir(results_dir):
        print(os.listdir(results_dir))
        shutil.rmtree(results_dir)
        os.makedirs(results_dir)
    detect.detection(weight_path=detect_weights_path, img_dir=linear_dir)
elif mode == 'detection':
    detect.detection(weight_path=detect_weights_path, img_dir=detection_dir)
else:
    raise NameError('Wrong MODE! The mode should be \'linear\' or \'detection\'')
