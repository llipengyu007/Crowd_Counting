import math


import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2
import numpy as np
import PIL
from numpy import ndarray
from PIL import Image, ImageOps

from .cc_model import HRNetCrowdCounting

class CrowdCountingInference(nn.Module):

    def __init__(self, model_root, device, **kwargs):
        super(CrowdCountingInference, self).__init__()

        assert isinstance(model_root, str), 'model must be a single str'

        self.device = device
        print(f'loading model from dir {model_root}')
        self.infer_model = HRNetCrowdCounting(model_root).to(self.device)
        self.infer_model.eval()
        print('load model done')

    def resize(self, img):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if resize_width >= 2048:
            tmp = resize_width
            resize_width = 2048
            resize_height = (resize_width / tmp) * resize_height

        if resize_height >= 2048:
            tmp = resize_height
            resize_height = 2048
            resize_width = (resize_height / tmp) * resize_width

        if resize_height <= 416:
            tmp = resize_height
            resize_height = 416
            resize_width = (resize_height / tmp) * resize_width
        if resize_width <= 416:
            tmp = resize_width
            resize_width = 416
            resize_height = (resize_width / tmp) * resize_height

        # other constraints
        if resize_height < resize_width:
            if resize_width / resize_height > 2048 / 416:  # 1024/416=2.46
                resize_width = 2048
                resize_height = 416
        else:
            if resize_height / resize_width > 2048 / 416:
                resize_height = 2048
                resize_width = 416

        resize_height = math.ceil(resize_height / 32) * 32
        resize_width = math.ceil(resize_width / 32) * 32
        img = transforms.Resize([resize_height, resize_width])(img)
        return img

    def merge_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(
                    eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(
                    3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (
                        i - 1)
                pred_w = math.floor(
                    3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (
                        j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w
                       + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,
                                            start_h:start_h + valid_h,
                                            start_w:start_w + valid_w]
        return pred_m

    def preprocess(self, input):
        img = self.convert_to_img(input)
        img = self.resize(img)
        img_ori_tensor = transforms.ToTensor()(img)
        img_shape = img_ori_tensor.shape
        img = transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))(
                                       img_ori_tensor)
        patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
        imgs = []
        for i in range(3):
            for j in range(3):
                start_h, start_w = (patch_height // 2) * i, (patch_width
                                                             // 2) * j
                imgs.append(img[:, start_h:start_h + patch_height,
                                start_w:start_w + patch_width])

        imgs = torch.stack(imgs)
        eval_img = imgs.to(self.device)
        eval_patchs = torch.squeeze(eval_img)
        prediction_map = torch.zeros(
            (1, 1, img_shape[1] // 2, img_shape[2] // 2)).to(self.device)
        result = {
            'img': eval_patchs,
            'map': prediction_map,
        }
        return result


    @torch.no_grad()
    def perform_inference(self, data):
        eval_patchs = data['img']
        prediction_map = data['map']
        eval_prediction, _, _ = self.infer_model(eval_patchs)
        eval_patchs_shape = eval_prediction.shape
        prediction_map = self.merge_crops(eval_patchs_shape, eval_prediction,
                                          prediction_map)

        return torch.sum(
            prediction_map, dim=(
                1, 2,
                3)).data.cpu().numpy(), prediction_map.data.cpu().numpy()[0][0]

    def convert_to_img(self, input):
        if isinstance(input, str):
            img = Image.open(input).convert("RGB")
        elif isinstance(input, PIL.Image.Image):
            img = input.convert('RGB')
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            img = input[:, :, ::-1]
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')
        return img

    def numpy_to_cv2img(self, img_array):
        """to convert a np.array with shape(h, w) to cv2 img

        Args:
            img_array (np.array): input data

        Returns:
            cv2 img
        """
        img_array = (img_array - img_array.min()) / (
                img_array.max() - img_array.min() + 1e-5)
        img_array = (img_array * 255).astype(np.uint8)
        img_array = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
        return img_array

    def alpha_img(self, img_src_array, img_heatmap_array, alpha=0.3):
        """get an alpha img by fusion source img and heatmap

        Args:
            img_src_array (np.array): input source data
            img_heatmap_array(np.array): heat map data
            alpha: hyper-parameter for fusion

        Returns:
            cv2 img
        """


        h,w = img_src_array.shape[:-1]
        img_heatmap_reshape_array = cv2.resize(img_heatmap_array, (w,h))
        output_array = cv2.addWeighted(img_heatmap_reshape_array, alpha, img_src_array, (1-alpha), 0)

        return output_array

    def __call__(self, img, *args, **kwargs):
        input = self.preprocess(img)
        counts, img_data = self.perform_inference(input)
        return {'Count': counts, 'HeatMap': img_data}
