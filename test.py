import os

import numpy as np
import scipy.io as scio
from tqdm import tqdm
import cv2
import time
from models.infer import CrowdCountingInference


img_root = r"./datasets/"
img_list = open("./datasets/imagelist.txt").readlines()
alpha = 0.5
device = "cuda:0"
#device = "cpu"
sign_visualize = True

crowd_counting_infer = CrowdCountingInference("./params/", device)

crowd_counting_infer.eval()

mae = 0
time_model = 0
time_postprocess = 0
avg_size = np.zeros(2)
for i, img in tqdm(enumerate(img_list)):
    img_path = os.path.join(img_root, img.strip())
    start_time = time.time()
    results = crowd_counting_infer(img_path)
    time_model += time.time() - start_time
    label = scio.loadmat(img_path.replace(".jpg",".mat"))
    error = abs(results["Count"][0] - int(len(label["annPoints"])))
    mae += error

    if sign_visualize:
        save_path = os.path.join('heatmap_res_imgs', img.strip())
        path, _ = os.path.split(save_path)
        if os.path.exists(path) == False:
            os.makedirs(path)
        start_time = time.time()
        heatmap_img = crowd_counting_infer.numpy_to_cv2img(results["HeatMap"])
        time_postprocess += time.time() - start_time
        cv2.imwrite(save_path, heatmap_img)

        save_path = os.path.join('alpha_res_imgs', img.strip())
        path, _ = os.path.split(save_path)
        if os.path.exists(path) == False:
            os.makedirs(path)
        img = cv2.imread(img_path)
        avg_size += np.array(img.shape[:2]).reshape(-1)

        start_time = time.time()
        alpha_img = crowd_counting_infer.alpha_img(img, heatmap_img, alpha)
        time_postprocess += time.time() - start_time
        cv2.imwrite(save_path, alpha_img)

print("MAE: ", mae/len(img_list))
print("Model Running Speed {:.4f}ms/img".format(time_model/len(img_list) * 1000))
print("Post Processing Speed {:.4f}ms/img".format(time_postprocess/len(img_list) * 1000))
print("Img Avg Size {}".format(avg_size/len(img_list)))

print("Complete")

