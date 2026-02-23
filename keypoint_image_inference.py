import cv2
import torch
import timm

from models.keypoint_head import CourtKeypointHead
from utils.heatmap_to_keypoints import heatmaps_to_keypoints



def predict_image(img_path=None, output_path=None):
    """
    Run keypoint detection on image.
    
    Args:
        video_source: 0 for webcam, or path to video file
        output_path: Optional path to save output video
    """

    if cv2.imread(img_path) is None:
        print('Invalid image path')
        return 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your model
    backbone = timm.create_model('mobilevit_s', pretrained=False, features_only=True)
    backbone.to(device)
    
    # Load checkpoint
    checkpoint = torch.load("weights/fine_tuning_keypoints_final.pth", map_location=device)
    
    # Get stride
    with torch.no_grad():
        x = torch.zeros(1, 3, 720, 1280).to(device)
        f = backbone(x)[-1]
        stride = x.shape[-1] // f.shape[-1]
    
    head = CourtKeypointHead(in_channels=f.shape[1], num_keypoints=14)
    head.to(device)
    
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    
    backbone.eval()
    head.eval()    
    
    img = cv2.imread(img_path)

    # inference
    with torch.no_grad():

        height, width = img.shape[0], img.shape[1]

        # preprocces image the same as the CourtKeypointDataset
        img_resized = cv2.resize(img, (1280, 720))  # width, height
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        input_tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .float() / 255.0
        ).unsqueeze(0).to(device)


        # Run inference
        features = backbone(input_tensor)
        pred_heatmaps = head(features[-1])
        
        # Convert to keypoints
        img_size = (720, 1280)
        pred_keypoints = heatmaps_to_keypoints(pred_heatmaps, img_size, stride)
        
        # Scale keypoints back to original frame size
        scale_x = width / 1280
        scale_y = height / 720
        
        # Draw keypoints on frame
        for (x, y) in pred_keypoints[0]:
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            cv2.circle(img, (x_scaled, y_scaled), 5, (0, 0, 255), -1)

        # draw lines
        img = draw_court_lines(img, pred_keypoints[0], scale_x, scale_y)


    cv2.imshow('Court Keypoint Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
def draw_court_lines(img, keypoints, scale_x, scale_y):
    """Draw lines connecting court keypoints"""
    connections = [
        (0, 4), (4, 6), (6, 1), (1, 3), (3, 7), (7, 5), (5 ,2), (2, 0), # doubles sideline/baseline
        (4, 6), (6, 9), (9, 11), (11, 7), (7, 5), (5, 10), (10, 8), (8, 4), # singles sideline/baseline
        (8, 12), (12, 9), (10, 13), (13, 11), (13, 12)  # service/center lines
    ]
    
    for (start_idx, end_idx) in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img




if __name__ == "__main__":
    # img_path = r"keypoint_data\images\_7UfL2egoN0_700.png"
    img_path = r"frames\theim_image.png"
    predict_image(img_path)

