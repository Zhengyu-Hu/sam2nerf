import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
import shutil

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10") # tab10 is a good color map for visualization
        cmap_idx = 0 if obj_id is None else obj_id # choose a color from tab10 based on the object id
        color = np.array([*cmap(cmap_idx)[:3], 0.6]) # 0.6 is the alpha value
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def png2jpeg(png_file, save_dir):
    img = Image.open(png_file)
    img = img.convert('RGB')
    os.makedirs(save_dir, exist_ok=True)
    img.save(os.path.join(save_dir, os.path.basename(png_file).replace('.png', '.jpeg')), quality=100)

def batch_png2jpeg(pngs_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for png_file in os.listdir(pngs_dir):
        if png_file.endswith('.png'):
            png2jpeg(os.path.join(pngs_dir, png_file), save_dir)

def circular_rename(video_dir, selected_idx, suffix=('jpeg', 'jpg', 'JPG', 'JPEG')):
    """
    Rename images in the directory to make selected image as 0.jpeg
    Args:
        video_dir: Directory containing the images
        selected_idx: Index of the image to become 0.jpeg
        suffix: Tuple of valid image extensions
    """
    # Get sorted list of image files
    frame_names = [f for f in sorted(os.listdir(video_dir)) 
                  if f.endswith(suffix)]
    
    # Create temporary directory for renaming
    temp_dir = os.path.join(video_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    total_frames = len(frame_names)
    
    # Copy files to temp directory with new names
    for i in range(total_frames):
        old_idx = (i + selected_idx) % total_frames
        new_name = f"{i:03}.jpeg" # zero-fill
        old_path = os.path.join(video_dir, frame_names[old_idx])
        temp_path = os.path.join(temp_dir, new_name)
        shutil.copy2(old_path, temp_path)
    
    return temp_dir

def make_tmp_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def split_for_back(video_dir, seg_idx, suffix=('jpeg', 'jpg', 'JPG', 'JPEG')):
    frame_names = [
        f for f in sorted(os.listdir(video_dir)) if f.endswith(suffix)
    ]
    total_frames = len(frame_names)

    forward_dir = os.path.join(video_dir,'forward_frames')
    backward_dir = os.path.join(video_dir,'backward_frames')
    make_tmp_dir(forward_dir)
    make_tmp_dir(backward_dir)

    for i in range(seg_idx,total_frames):
        source = os.path.join(video_dir, frame_names[i])
        destination = os.path.join(forward_dir, f'{i:03}.jpeg')
        shutil.copy(source, destination)

    for i in range(seg_idx+1):
        source = os.path.join(video_dir, frame_names[i])
        new_idx = seg_idx-i
        destination = os.path.join(backward_dir, f'{new_idx:03}.jpeg')
        shutil.copy(source, destination)
    
    return forward_dir, backward_dir


if __name__ == '__main__':
    # Convert png to jpeg
    """ pngs_dir = 'render_result/lego'
    save_dir = 'render_result/lego-jpeg'
    batch_png2jpeg(pngs_dir, save_dir) """
    video_dir = 'render_result/lego-jpeg'
    suffix = ('jpeg', 'jpg', 'JPG', 'JPEG')
    seg_idx = 7
    split_for_back(video_dir, seg_idx)

   