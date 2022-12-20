import os
import scipy.io as scio
from tqdm import tqdm
import cv2

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
for i, img in tqdm(enumerate(img_list)):
    img_path = os.path.join(img_root, img.strip())
    results = crowd_counting_infer(img_path)
    label = scio.loadmat(img_path.replace(".jpg",".mat"))
    error = abs(results["Count"][0] - int(len(label["annPoints"])))
    mae += error

    if sign_visualize:
        save_path = os.path.join('heatmap_res_imgs', img.strip())
        path, _ = os.path.split(save_path)
        if os.path.exists(path) == False:
            os.makedirs(path)
        heatmap_img = crowd_counting_infer.numpy_to_cv2img(results["HeatMap"])
        cv2.imwrite(save_path, heatmap_img)

        save_path = os.path.join('alpha_res_imgs', img.strip())
        path, _ = os.path.split(save_path)
        if os.path.exists(path) == False:
            os.makedirs(path)
        img = cv2.imread(img_path)
        alpha_img = crowd_counting_infer.alpha_img(img, heatmap_img, alpha)
        cv2.imwrite(save_path, alpha_img)

print("MAE: ", mae/len(img_list))

print("Complete")
