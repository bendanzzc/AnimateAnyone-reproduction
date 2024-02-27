import os
import torch
import datetime
import numpy as np
from PIL import Image
from pipeline_stable_video_diffusion_controlnet_long import StableVideoDiffusionPipelineControlNet
from controlnet_sdv import ControlNetSDVModel
#from diffusers import T2IAdapter
from unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re 

def write_mp4(video_path, samples):
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"MP4V"),
        15,
        (samples[0].shape[1], samples[0].shape[0]),
    )

    for frame in samples:
        writer.write(frame)
    writer.release()

def save_gifs_side_by_side(batch_output, validation_images, validation_control_images, output_folder):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    video_path = output_folder+'/test_1.mp4'
    final_images = []
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    for idx, image_list in enumerate(zip(validation_images, validation_control_images, flattened_batch_output)):
        result = get_concat_h(image_list[0], image_list[1])
        result = get_concat_h(result, image_list[2])
        final_images.append(np.array(result)[:,:,::-1])
    write_mp4(video_path, final_images)

# Define functions
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(448, 768)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def tensor_to_pil(tensor):
    """ Convert a PyTorch tensor to a PIL Image. """
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # Flatten batch_output to a list of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Convert tensors in lists to PIL Images
    validation_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images]
    validation_control_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_control_images]
    flattened_batch_output = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output]

    # Flatten lists if they contain sublists (for tensors converted to multiple images)
    validation_images = [img for sublist in validation_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    validation_control_images = [img for sublist in validation_control_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_batch_output = [img for sublist in flattened_batch_output for img in (sublist if isinstance(sublist, list) else [sublist])]

    # Combine frames into a list
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols

    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols, target_size=(256, 256))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        matches = re.findall(r'\d+', filename)  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'


    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder))

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images

def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        matches = re.findall(r'\d+', filename)  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'


    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder))

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images[::2]

# Usage example
def convert_list_bgra_to_rgba(image_list):
    """
    Convert a list of PIL Image objects from BGRA to RGBA format.

    Parameters:
    image_list (list of PIL.Image.Image): A list of images in BGRA format.

    Returns:
    list of PIL.Image.Image: The list of images converted to RGBA format.
    """
    rgba_images = []
    for image in image_list:
        if image.mode == 'RGBA' or image.mode == 'BGRA':
            # Split the image into its components
            b, g, r, a = image.split()
            # Re-merge in RGBA order
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            # For non-alpha images, assume they are BGR and convert to RGB
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))

        rgba_images.append(converted_image)

    return rgba_images

# Main script
if __name__ == "__main__":
    from tqdm import tqdm
    args = {
        "pretrained_model_name_or_path": "SVD/svd_14",
        "validation_image_folder": "./testcase/81FyMPk-WIS/images",
        "validation_control_folder": "./testcase/81FyMPk-WIS/dwpose_woface",
        "output_dir": "./output",
        "height": 896,
        "width":704,
        # cant be bothered to add the args in myself, just use notepad
    }

    # Load validation images and control images
    validation_images = load_images_from_folder_to_pil(args["validation_image_folder"], (args['width'], args['height']))
    #validation_images = convert_list_bgra_to_rgba(validation_images)
    validation_control_images = load_images_from_folder_to_pil(args["validation_control_folder"], (args['width'], args['height']))


    controlnet = ControlNetSDVModel.from_pretrained("/checkpoint/controlnet")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"]+'/unet')
    
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args["pretrained_model_name_or_path"], controlnet=controlnet, unet=unet)
    pipeline.to(dtype=torch.float16)
    pipeline.enable_model_cpu_offload()

    val_save_dir = os.path.join(args["output_dir"], "validation_images")
    os.makedirs(val_save_dir, exist_ok=True)

    # Inference and saving loop
    final_result = []
    #ref_image = validation_images[0]
    ref_image = Image.open('./testcase/test_12.png').convert('RGB')
    frames = 14
    num_frames = len(validation_images)

    video_frames = pipeline(ref_image, validation_control_images[:num_frames], decode_chunk_size=2,num_frames=num_frames,motion_bucket_id=127.0, fps=7,controlnet_cond_scale=1.0, width=args['width'], height=args["height"], min_guidance_scale=3.5, max_guidance_scale=3.5, frames_per_batch=frames, num_inference_steps=25, overlap=4).frames[0]
    final_result.append(video_frames)

    save_gifs_side_by_side(final_result,validation_images[:num_frames], validation_control_images[:num_frames],val_save_dir)
