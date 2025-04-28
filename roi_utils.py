import torch
import torch.nn.functional as F

def create_roi_mask_from_bboxes(image_shape, bboxes, device):
    """
    Create an ROI mask from a list of bounding boxes.
    
    Args:
        image_shape (tuple): The shape of the image (batch_size, channels, height, width).
        bboxes (list): A list of bounding boxes, where each bounding box is a tuple (x1, y1, x2, y2).
        device (torch.device): The device to create the mask on.
    
    Returns:
        torch.Tensor: The ROI mask, with 1s for ROI regions and 0s elsewhere.
    """
    batch_size, _, height, width = image_shape
    roi_mask = torch.zeros((batch_size, 1, height, width), device=device)
    
    for i, bbox in enumerate(bboxes):
        if i >= batch_size:
            break
        
        x1, y1, x2, y2 = bbox
        roi_mask[i, 0, y1:y2, x1:x2] = 1.0
    
    return roi_mask

def encode_with_roi(model, image, bboxes, roi_weight=2.0):
    """
    Encode an image with ROI prioritization using bounding boxes.
    
    Args:
        model (DeepJSCC): The DeepJSCC model.
        image (torch.Tensor): The input image.
        bboxes (list): A list of bounding boxes, where each bounding box is a tuple (x1, y1, x2, y2).
        roi_weight (float): The weight to apply to ROI regions.
    
    Returns:
        torch.Tensor: The encoded image.
    """
    # Create ROI mask from bounding boxes
    roi_mask = create_roi_mask_from_bboxes(image.shape, bboxes, image.device)
    
    # Encode with ROI prioritization
    return model.encode_with_roi(image, roi_mask, roi_weight)