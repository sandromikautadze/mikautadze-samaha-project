from collections import deque
from tqdm import tqdm
import numpy as np
import supervision as sv
import torch
import os
from ultralytics import YOLO
from sports.common.team import TeamClassifier
from inference import get_model
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch, draw_pitch_voronoi_diagram
from sports.configs.soccer import SoccerPitchConfiguration
import cv2
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_crops(model, input_path, conf = 0.3, stride=30, is_yolov8 = False):
    """
    Extracts cropped images of detected objects from video frames.

    Args:
        model (YOLO): YOLO detection model used for extracting crops.
        input_path (str): Path to the input video file.
        conf (float, optional): Confidence threshold for the detection model. Defaults to 0.3.
        stride (int, optional): Frame stride for processing the video. Defaults to 30.

    Returns:
        list: List of cropped images from the video frames.
    """
    frame_generator = sv.get_video_frames_generator(input_path)
    video_info = sv.VideoInfo.from_video_path(input_path)
    crops = []
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Cropping..."):
        if is_yolov8 is False:
            results = model(frame, conf = conf, verbose=False, vid_stride=stride)[0] # ideally stride is set to be 1 crop per second of the video
            detections = sv.Detections.from_ultralytics(results)
        else:
            results = model.infer(frame, conf = conf, verbose=False, vid_stride=stride)[0] # ideally stride is set to be 1 crop per second of the video
            detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == 2] # players
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    return crops

def get_goalkeepers_team_id(players_detection, goalkeepers_detection):
    """
    Determines the team IDs of detected goalkeepers based on proximity to detected players.

    Args:
        players_detection (sv.Detections): Detections of players in the frame.
        goalkeepers_detection (sv.Detections): Detections of goalkeepers in the frame.

    Returns:
        numpy.ndarray: Array of team IDs corresponding to each detected goalkeeper.
    """
    
    goalkeepers_detection_xy = goalkeepers_detection.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_detection_xy = players_detection.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Handle empty arrays
    team0_mean = players_detection_xy[players_detection.class_id == 0].mean(axis=0) if np.any(players_detection.class_id == 0) else np.array([0, 0])
    team1_mean = players_detection_xy[players_detection.class_id == 1].mean(axis=0) if np.any(players_detection.class_id == 1) else np.array([0, 0])
    
    goalkeepers_team_id = []
    for goalkeeper_detection_xy in goalkeepers_detection_xy:
        dist_0 = np.linalg.norm(goalkeeper_detection_xy - team0_mean)
        dist_1 = np.linalg.norm(goalkeeper_detection_xy - team1_mean)
        team_id = 0 if dist_0 < dist_1 else 1
        goalkeepers_team_id.append(team_id)
        
    return np.array(goalkeepers_team_id)

def infer_video_and_save_pitch_images(
    player_model, field_model, input_video_path, pitch_configuration,
    team1_color = sv.Color.RED, team2_color = sv.Color.BLUE, ref_color = sv.Color.YELLOW, ball_color = sv.Color.WHITE,
    conf_player=0.3, conf_pitch = 0.3, conf_keypoints = 0.5, nsm_threshold = 0.5, maxlen = 5, device = "cuda", output_base_folder = '.',
    is_yolov8 = False,
):
    """
    Processes a video to detect and track objects, annotate frames, and generate pitch-related visualizations.

    Args:
        player_model (YOLO): YOLO model used for detecting players.
        field_model (Model): Model used for detecting pitch keypoints.
        input_video_path (str): Path to the input video file.
        pitch_configuration (SoccerPitchConfiguration): Configuration of the pitch layout.
        team1_color (object, optional): Color for team 1 annotations. Defaults to sv.Color.RED.
        team2_color (object, optional): Color for team 2 annotations. Defaults to sv.Color.BLUE.
        ref_color (object, optional): Color for referee annotations. Defaults to sv.Color.YELLOW.
        ball_color (object, optional): Color for ball annotations. Defaults to sv.Color.WHITE.
        conf_player (float, optional): Confidence threshold for player detection. Defaults to 0.3.
        conf_pitch (float, optional): Confidence threshold for pitch keypoint detection. Defaults to 0.3.
        conf_keypoints (float, optional): Confidence threshold for filtering keypoints. Defaults to 0.5.
        nsm_threshold (float, optional): Non-maximum suppression threshold. Defaults to 0.5.
        maxlen (int, optional): Maximum length of the deque for averaging transformations. Defaults to 5.
        device (str, optional): Device for model inference. Defaults to "cuda".
        output_base_folder (str, optional): Base folder to save outputs. Defaults to current directory (".").
    """
    # CREATE FOLDERS IF NECESSARY
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_folder = os.path.join(output_base_folder, f"{base_name}_inference")
    frames_output_folder = os.path.join(output_folder, "frames")
    video_output_path = os.path.join(output_folder, "players.mp4")
    pitch_output_folder = os.path.join(output_folder, "frames_pitch")
    voronoi_output_folder = os.path.join(output_folder, "frames_voronoi")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(frames_output_folder, exist_ok=True)
    os.makedirs(pitch_output_folder, exist_ok=True)
    os.makedirs(voronoi_output_folder, exist_ok=True)

    # ANNOTATORS
    ball_annotator = sv.TriangleAnnotator(color=ball_color, base=18, height=15)
    others_annotator = sv.TriangleAnnotator(color=sv.ColorPalette([team1_color, team2_color, ref_color]), base=20, height=17)

    # TRACKER
    tracker = sv.ByteTrack()
    tracker.reset()
    
    # TEAM CLASSIFIER
    crops = get_crops(player_model, input_video_path, conf_player, is_yolov8=is_yolov8)
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # INFO TO SET UP GENERATION
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    video_sink = sv.VideoSink(video_output_path, video_info=video_info)
    frame_generator = sv.get_video_frames_generator(input_video_path)
    
    M = deque(maxlen=maxlen)
    
    with video_sink:
        for frame_idx, frame in tqdm(enumerate(frame_generator), total=video_info.total_frames, desc="Processing"):            
            
            # SAVE FRAME AS PIC
            frame_path = os.path.join(frames_output_folder, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
                        
            #################
            ### DETECTION ###
            #################
            if is_yolov8 is False:
                results = player_model(frame, conf=conf_player, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
            else:
                results = player_model.infer(frame, conf=conf_player, verbose=False)[0]
                detections = sv.Detections.from_inference(results)
            
            ## BALL DETECTION
            ball_detections = detections[detections.class_id == 0]
            ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, 10)

            ## OTHERS' DETECTIONS (player, goalkeeper, referee)
            others_detections = detections[detections.class_id != 0]
            others_detections = others_detections.with_nms(threshold=nsm_threshold, class_agnostic=True)
            others_detections = tracker.update_with_detections(others_detections)

            ### SEPARATE DETECTIONS OF CLASSES
            players_detections = others_detections[others_detections.class_id == 2]  #### PLAYERS
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(player_crops)
            goalkeepers_detections = others_detections[others_detections.class_id == 1]  #### GOALKEEPERS
            goalkeepers_detections.class_id = get_goalkeepers_team_id(players_detections, goalkeepers_detections)
            refs_detections = others_detections[others_detections.class_id == 3]  #### REFS
            refs_detections.class_id -= 1  # adjust class_id so that refs become a separate class index

            ## MERGE BACK THE DETECTIONS
            others_detections = sv.Detections.merge([players_detections, goalkeepers_detections, refs_detections])
            others_detections.class_id = others_detections.class_id.astype(int)

            ## ANNOTATION
            annotated_frame = frame.copy()
            annotated_frame = ball_annotator.annotate(annotated_frame, detections=ball_detections)
            annotated_frame = others_annotator.annotate(annotated_frame, detections=others_detections)
            
            #############
            ### PITCH ###
            #############
            result = field_model.infer(frame, confidence=conf_pitch)[0]
            key_points = sv.KeyPoints.from_inference(result)
            filter_mask = key_points.confidence[0] > conf_keypoints
            frame_reference_points = key_points.xy[0][filter_mask]
            pitch_reference_points = np.array(pitch_configuration.vertices)[filter_mask]

            view_transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )
            
            M.append(view_transformer.m)
            view_transformer.m = np.mean(np.array(M), axis=0) # average out the keypoints to remove the little fluctuations

            # Transform coordinates
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = view_transformer.transform_points(frame_ball_xy)
            frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = view_transformer.transform_points(frame_players_xy)
            frame_refs_xy = refs_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_refs_xy = view_transformer.transform_points(frame_refs_xy)

            ## DRAW PITCH WITH PLAYERS
            pitch = draw_pitch(config=pitch_configuration)
            pitch = draw_points_on_pitch(
                config=pitch_configuration, xy=pitch_players_xy[players_detections.class_id == 0],
                face_color=team1_color, edge_color=sv.Color.BLACK,
                radius=11, pitch=pitch
            )
            pitch = draw_points_on_pitch(
                config=pitch_configuration, xy=pitch_players_xy[players_detections.class_id == 1],
                face_color=team2_color, edge_color=sv.Color.BLACK,
                radius=11, pitch=pitch
            )
            pitch = draw_points_on_pitch(
                config=pitch_configuration, xy=pitch_refs_xy,
                face_color=ref_color, edge_color=sv.Color.BLACK,
                radius=11, pitch=pitch
            )
            pitch = draw_points_on_pitch(
                config=pitch_configuration, xy=pitch_ball_xy,
                face_color=ball_color, edge_color=sv.Color.BLACK,
                radius=9, pitch=pitch
            )
            # SAVE PITCH WITH PLAYERS
            pitch_frame_path = os.path.join(pitch_output_folder, f"pitch_{frame_idx:04d}.jpg")
            cv2.imwrite(pitch_frame_path, pitch)
            
            # DRAW VORONOI DIAGRAM
            pitch_voronoi = draw_pitch(
                config=pitch_configuration,
                background_color=sv.Color.WHITE,
                line_color=sv.Color.BLACK
            )
            pitch_voronoi = draw_pitch_voronoi_diagram(
                config=pitch_configuration,
                team_1_xy=pitch_players_xy[players_detections.class_id == 0],
                team_2_xy=pitch_players_xy[players_detections.class_id == 1],
                team_1_color=team1_color, team_2_color=team2_color,
                pitch=pitch_voronoi,
            )
            pitch_voronoi = draw_points_on_pitch(config=pitch_configuration,
                xy=pitch_ball_xy,
                face_color=sv.Color.WHITE, edge_color=sv.Color.WHITE,
                radius=8, thickness=1,
                pitch=pitch_voronoi,
            )
            pitch_voronoi = draw_points_on_pitch(config=pitch_configuration,
                xy=pitch_players_xy[players_detections.class_id == 0],
                face_color=team1_color, edge_color=sv.Color.BLACK,
                radius=16, thickness=1,
                pitch=pitch_voronoi,
            )
            pitch_voronoi = draw_points_on_pitch(config=pitch_configuration,
                xy=pitch_players_xy[players_detections.class_id == 1],
                face_color=team2_color, edge_color=sv.Color.BLACK,
                radius=16, thickness=1,
                pitch=pitch_voronoi,
            )
            
            # SAVE VORONOI DIAGRAM
            
            pitch_voronoi_path = os.path.join(voronoi_output_folder, f"voronoi_{frame_idx:04d}.jpg")
            cv2.imwrite(pitch_voronoi_path, pitch_voronoi)
            
            # SAVE FRAME TO VIDEO
            video_sink.write_frame(annotated_frame)

    print(f"All frames saved in: {frames_output_folder}")
    print(f"All pitch frames saved in: {pitch_output_folder}")
    print(f"All voronoi frames saved in: {voronoi_output_folder}")
    print(f"Annotated video saved at: {video_output_path}")
    
def process_offside_detection(frame_path, player_model, field_model, right_team, left_team, CONFIG, attacking_team="right",
                              team1_color=sv.Color.BLUE, team2_color=sv.Color.RED, ref_color=sv.Color.YELLOW, ball_color=sv.Color.WHITE,
                              conf_player=0.3, conf_pitch=0.3, conf_keypoints=0.5, nsm_threshold=0.5, device="cuda", is_yolov8=False):
    """
    Detects and visualizes offside lines in a football match frame and generates an annotated pitch image.

    Args:
        frame_path (str): Path to the image frame to process.
        player_model (YOLO): YOLO model used for detecting players in the frame.
        field_model (Model): Model used for detecting pitch keypoints.
        right_team (int): ID of the team positioned on the right side of the pitch.
        left_team (int): ID of the team positioned on the left side of the pitch.
        CONFIG (SoccerPitchConfiguration): Configuration object containing pitch dimensions and keypoints.
        attacking_team (str, optional): The team currently attacking ("right" or "left"). Defaults to "right".
        team1_color (sv.Color, optional): Color for team 1 annotations. Defaults to sv.Color.BLUE.
        team2_color (sv.Color, optional): Color for team 2 annotations. Defaults to sv.Color.RED.
        ref_color (sv.Color, optional): Color for referee annotations. Defaults to sv.Color.YELLOW.
        ball_color (sv.Color, optional): Color for ball annotations. Defaults to sv.Color.WHITE.
        conf_player (float, optional): Confidence threshold for player detection. Defaults to 0.3.
        conf_pitch (float, optional): Confidence threshold for pitch keypoint detection. Defaults to 0.3.
        conf_keypoints (float, optional): Confidence threshold for filtering detected keypoints. Defaults to 0.5.
        nsm_threshold (float, optional): Non-maximum suppression threshold. Defaults to 0.5.
        device (str, optional): Device for model inference (e.g., "cuda" or "cpu"). Defaults to "cuda".
        is_yolov8 (bool, optional): Indicates whether the model is YOLOv8. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - pitch (np.ndarray): Annotated 2D pitch image with offside lines and players.
            - annotated_frame (np.ndarray): Annotated original frame with offside visualization.
    """

    # Load the video frame
    image_frame = cv2.imread(frame_path)
    
    # ANNOTATORS
    ball_annotator = sv.TriangleAnnotator(color=ball_color, base=18, height=15)
    others_annotator = sv.TriangleAnnotator(color=sv.ColorPalette([team1_color, team2_color, ref_color]), base=20, height=17)
    
    crops = get_crops(player_model, frame_path, 0.5, is_yolov8=is_yolov8)
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    if is_yolov8 is False:
        results = player_model(image_frame, conf=conf_player, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
    else:
        results = player_model.infer(image_frame, conf=conf_player, verbose=False)[0]
        detections = sv.Detections.from_inference(results)
    
    # # ball detections
    ball_detections = detections[detections.class_id == 0]
    ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, 10)  # add 10 pixels to box so that triangles are above
  
  ## OTHERS' DETECTIONS (player, goalkeeper, referee)
    others_detections = detections[detections.class_id != 0]
    others_detections = others_detections.with_nms(threshold=nsm_threshold, class_agnostic=True)
    # others_detections = tracker.update_with_detections(others_detections)

    ### SEPARATE DETECTIONS OF CLASSES
    players_detections = others_detections[others_detections.class_id == 2]  #### PLAYERS
    player_crops = [sv.crop_image(image_frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(player_crops)
    goalkeepers_detections = others_detections[others_detections.class_id == 1]  #### GOALKEEPERS
    goalkeepers_detections.class_id = get_goalkeepers_team_id(players_detections, goalkeepers_detections)
    refs_detections = others_detections[others_detections.class_id == 3]  #### REFS
    refs_detections.class_id -= 1  # adjust class_id so that refs become a separate class index
    
    ## MERGE BACK THE DETECTIONS
    others_detections = sv.Detections.merge([players_detections, goalkeepers_detections, refs_detections])
    others_detections.class_id = others_detections.class_id.astype(int)

    ## ANNOTATION
    annotated_frame = image_frame.copy()
    annotated_frame = ball_annotator.annotate(annotated_frame, detections=ball_detections)
    annotated_frame = others_annotator.annotate(annotated_frame, detections=others_detections)
           
    
    # Keypoint detection and homography
    result = field_model.infer(image_frame, confidence=conf_pitch)[0]
    key_points = sv.KeyPoints.from_inference(result)
    filter = key_points.confidence[0] > conf_keypoints
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]
    view_transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
    
    view_transformer2 = ViewTransformer(source=pitch_reference_points, target=frame_reference_points)
    
    # Transform coordinates to the 2D pitch
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = view_transformer.transform_points(frame_ball_xy)
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = view_transformer.transform_points(frame_players_xy)
    frame_refs_xy = refs_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_refs_xy = view_transformer.transform_points(frame_refs_xy)

    # Initialize the pitch
    pitch = draw_pitch(config=CONFIG)
    pitch = draw_points_on_pitch(
        config=CONFIG, xy=pitch_players_xy[players_detections.class_id == 0],
        face_color=team1_color, edge_color=sv.Color.BLACK,
        radius=11, pitch=pitch
    )
    pitch = draw_points_on_pitch(
        config=CONFIG, xy=pitch_players_xy[players_detections.class_id == 1],
        face_color=team2_color, edge_color=sv.Color.BLACK,
        radius=11, pitch=pitch
    )
    pitch = draw_points_on_pitch(
        config=CONFIG, xy=pitch_refs_xy,
        face_color=ref_color, edge_color=sv.Color.BLACK,
        radius=11, pitch=pitch
    )
    pitch = draw_points_on_pitch(
        config=CONFIG, xy=pitch_ball_xy,
        face_color=ball_color, edge_color=sv.Color.BLACK,
        radius=9, pitch=pitch
    )

    # Determine attacking and defending players
    centerline_x = CONFIG.length / 2
    if attacking_team == "right":
        defending_players = pitch_players_xy[players_detections.class_id == left_team]
        attacking_players = pitch_players_xy[players_detections.class_id == right_team]
    else:
        defending_players = pitch_players_xy[players_detections.class_id == right_team]
        attacking_players = pitch_players_xy[players_detections.class_id == left_team]

    if len(defending_players) > 0 and len(attacking_players) > 0:
        # Calculate offside and attacker lines
        last_defender_x = np.min(defending_players[:, 0]) if attacking_team == "right" else np.max(defending_players[:, 0])
        offside_x = min(last_defender_x, centerline_x) if attacking_team == "right" else max(last_defender_x, centerline_x)

        last_attacker_x = (np.min(attacking_players[:, 0]) if attacking_team == "right"
                        else np.max(attacking_players[:, 0]))
        last_attacker_x = min(last_attacker_x, offside_x) if attacking_team == "right" else max(last_attacker_x, offside_x)

        # Draw the offside and attacker lines
        offside_line = np.array([[offside_x, 0], [offside_x, CONFIG.width]])
        attacker_line = np.array([[last_attacker_x, 0], [last_attacker_x, CONFIG.width]])

        grey_color = sv.Color(128, 128, 128)
        black_color = sv.Color(0, 0, 0)
        pitch = draw_paths_on_pitch(config=CONFIG, paths=[offside_line], color=grey_color, pitch=pitch)
        pitch = draw_paths_on_pitch(config=CONFIG, paths=[attacker_line], color=black_color, pitch=pitch)

        # Highlight offside area
        offside_area = np.array([
            [offside_x, 0], [offside_x, CONFIG.width],
            [last_attacker_x, CONFIG.width], [last_attacker_x, 0]
        ])
        pitch = cv2.fillPoly(pitch, [np.array(offside_area, dtype=np.int32)], (200, 200, 200))

        # Map to original frame
        offside_line_frame = view_transformer2.transform_points(offside_line)
        attacker_line_frame = view_transformer2.transform_points(attacker_line)
        offside_area_frame = view_transformer2.transform_points(offside_area)
        annotated_frame = image_frame.copy()
        annotated_frame = cv2.polylines(annotated_frame, [offside_line_frame.astype(np.int32)], False, (128, 128, 128), 2)
        annotated_frame = cv2.polylines(annotated_frame, [attacker_line_frame.astype(np.int32)], False, (0, 0, 0), 2)
        annotated_frame = cv2.fillPoly(annotated_frame, [offside_area_frame.astype(np.int32)], (200, 200, 200))
         
    return pitch, annotated_frame

