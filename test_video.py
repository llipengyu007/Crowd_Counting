import os
import cv2

from models.infer import CrowdCountingInference

video_root = "./datasets/"
video_name = "video_sample.mp4"
video_path = os.path.join(video_root, video_name)
alpha = 0.5
device = "cuda:0"
#device = "cpu"

crowd_counting_infer = CrowdCountingInference("./params/", device)
crowd_counting_infer.eval()


cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

save_path = os.path.join("alpha_res_imgs", video_name+".avi")
path, _ = os.path.split(save_path)
if os.path.exists(path) == False:
    os.mkdir(path)
print("save path:{}".format(save_path))
videowriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w,h))

ret = True
ind = 0
while(ret):
    ret, frame = cap.read()
    if ret:
        results = crowd_counting_infer(frame)

        heatmap_frame = crowd_counting_infer.numpy_to_cv2img(results["HeatMap"])
        alpha_frame = crowd_counting_infer.alpha_img(frame, heatmap_frame, alpha)
        videowriter.write(alpha_frame)
    if ind % fps == 0:
        print("running: {} / {}".format(ind, num_frame))
    ind += 1


print("Complete")

