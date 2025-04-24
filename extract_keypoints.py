import cv2
import mediapipe as mp
import numpy as np

def extract_keypoints_from_video(video_path, max_frames=60):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_seq = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
            keypoints_seq.append(keypoints)
        else:
            keypoints_seq.append(np.zeros((33, 4)))
        frame_count += 1

    cap.release()
    pose.close()
    while len(keypoints_seq) < max_frames:
        keypoints_seq.append(np.zeros((33, 4)))
    return np.array(keypoints_seq[:max_frames])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python extract_keypoints.py <video_path> <output.npy>")
        exit(1)
    video_path = sys.argv[1]
    output_path = sys.argv[2]
    keypoints = extract_keypoints_from_video(video_path)
    np.save(output_path, keypoints)
    print(f"Keypoints guardados en {output_path}")
