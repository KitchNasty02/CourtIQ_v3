import cv2
import torch
import yt_dlp
import timm
import os
import numpy as np

from models.keypoint_head import CourtKeypointHead
from utils.heatmap_to_keypoints import heatmaps_to_keypoints

MOVING_AVERAGE_NUM = 5  # 5 point moving average


def download_youtube_video(url, output_path="youtube_video.mp4"):
    """Download YouTube video"""
    ydl_opts = {
        # 'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best', # download in 1080x720
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return output_path


def calculate_keypoint_moving_average(pred_history, new_pred):

    # skip if not enough pred yet
    if len(pred_history) < MOVING_AVERAGE_NUM:
        pred_history.append(new_pred)
        return new_pred, pred_history
    
    pred_history.pop(0)     # remove oldest prediction
    pred_history.append(new_pred)

    avg_pred = np.mean(pred_history, axis=0)
    
    return avg_pred, pred_history


def predict_video(video_source=0, output_path=None):
    """
    Run keypoint detection on video.
    
    Args:
        video_source: 0 for webcam, or path to video file
        output_path: Optional path to save output video
    """
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
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Add this to your video processing code
    print(f"Training size: 1280x720")
    print(f"Video size: {width}x{height}")
    print(f"Aspect ratio - Training: {1280/720:.2f}, Video: {width/height:.2f}")
    
    # Setup video writer if saving output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {fps} FPS, {width}x{height}")
    
    frame_count = 0

    inference_history = []
    last_pred = 0
    inference_frame_skip = 5
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            if frame_count % inference_frame_skip != 0 and frame_count > 1:
                for (x, y) in last_pred:
                    x_scaled = int(x * scale_x)
                    y_scaled = int(y * scale_y)
                    cv2.circle(frame, (x_scaled, y_scaled), 5, (0, 0, 255), -1)
                
                # Draw court lines (optional - connect keypoints)
                draw_court_lines(frame, last_pred, scale_x, scale_y)
                
                # Display FPS
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if writer:
                    writer.write(frame)
                continue
            
            # preprocces image the same as the CourtKeypointDataset
            img_resized = cv2.resize(frame, (1280, 720))  # width, height
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
            img_size = (720, 1280)  # Your training size
            pred_keypoints = heatmaps_to_keypoints(pred_heatmaps, img_size, stride)
            

            # find 5 point moving average
            avg_pred, inference_history = calculate_keypoint_moving_average(inference_history, pred_keypoints[0])
            last_pred = inference_history[-1]
            
            # Scale keypoints back to original frame size
            scale_x = width / 1280
            scale_y = height / 720
            
            # Draw keypoints on frame
            for (x, y) in avg_pred:
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                cv2.circle(frame, (x_scaled, y_scaled), 5, (0, 0, 255), -1)
            
            # Draw court lines (optional - connect keypoints)
            draw_court_lines(frame, avg_pred, scale_x, scale_y)
            
            # Display FPS
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Court Keypoint Detection', frame)
            
            # Write to output
            if writer:
                writer.write(frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")


def draw_court_lines(frame, keypoints, scale_x, scale_y):
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
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)



def predict_youtube_video(youtube_url, output_path="output.mp4"):
    """
    Download YouTube video and run predictions
    """
    print("Downloading YouTube video...")
    video_path = download_youtube_video(youtube_url, "temp_video.mp4")
    
    print("Running predictions...")
    predict_video(video_source=video_path, output_path=output_path)
    
    # clean up
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")
    
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    # predict_video(video_source=0)
    
    # youtube_url = "https://www.youtube.com/watch?v=C4Gl-T2dtss"
    # predict_youtube_video(youtube_url, output_path="output/court_detection_Alcaraz_Novak.mp4")

    # youtube_url = "https://www.youtube.com/watch?v=I6b69yvtufI"
    # predict_youtube_video(youtube_url, output_path="output/court_detection_Alcaraz_Paul.mp4")

    # youtube_url = "https://www.youtube.com/watch?v=XOR1EuU-08A"
    # predict_youtube_video(youtube_url, output_path="output/court_detection_Federer_Ivashka.mp4")

    # youtube_url = "https://www.youtube.com/watch?v=OneBfXLVV60"
    # predict_youtube_video(youtube_url, output_path="output/court_detection_hit_with_John.mp4")

    youtube_url = "https://www.youtube.com/watch?v=0KlllgCrvWo"
    predict_youtube_video(youtube_url, output_path="output/court_detection_Novak_Zverev.mp4")



