import torch
from sam2.build_sam import build_sam2_video_predictor
from videos_helper import *
from matplotlib import pyplot as plt
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
sam2_ckpt = './checkpoint/sam2ckpts/sam2.1_hiera_base_plus.pt'
# BUG: Use the sam2/configs/sam2.1/sam2.1_hiera_b+.yaml in the folder of site-packages
# not the relative path of the working directory
model_cfg = './configs/sam2.1/sam2.1_hiera_b+.yaml'
predictor = build_sam2_video_predictor(model_cfg, sam2_ckpt,device)

# Load the video
video_dir = 'render_result/lego-jpeg'
suffix = ('jpeg', 'jpg', 'JPG', 'JPEG')

# Select which frames to segment
seg_idx = 7
# circular index
temp_dir = circular_rename(video_dir, seg_idx, suffix)
frame_names = [f for f in sorted(os.listdir(temp_dir)) 
          if f.endswith(suffix)]
""" plt.title(f'Frame {seg_idx} is chosen to segment')
img = Image.open(os.path.join(temp_dir, frame_names[0]))
plt.imshow(img)
plt.show() """
# Add positive clicks
obj_id = 2
points = np.array([[300, 125],[325, 160],[250,125]], dtype=np.float32)
labels = np.ones(points.shape[0])

#  Loads all the JPEG frames in video_path and stores their pixels in inference_state
inference_state = predictor.init_state(video_path=temp_dir)

# Segment and show the result
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
inference_state=inference_state,
frame_idx=0,
obj_id=obj_id,
points=points,
labels=labels,
)

mask = plt.figure()
plt.title(f'Frame {seg_idx} Segmentation Result')
plt.imshow(Image.open(os.path.join(temp_dir, frame_names[0])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
mask.savefig(f'Frame_{seg_idx}_seg_2Dmask.png')

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
'''
vis_frame_stride = 5
for out_frame_idx in range(seg_idx, len(frame_names), vis_frame_stride):
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()
'''

from matplotlib.animation import FuncAnimation
fig = plt.figure()
def init():
    plt.cla()
def show_seg_frame(frame_idx):
    plt.imshow(Image.open(os.path.join(temp_dir, frame_names[frame_idx])))
    for out_obj_id, out_mask in video_segments[frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
frame = np.arange(0, len(frame_names))
ani = FuncAnimation(fig, show_seg_frame, frames=frame, init_func=init)
ani.save("seg_video.gif", fps=20)