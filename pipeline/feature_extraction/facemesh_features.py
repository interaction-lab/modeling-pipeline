
import pandas as pd
from pipeline.common.file_utils import ensure_destination_exists
import cv2
import mediapipe as mp


def get_facemesh_features(srv_video: str, dst_csv: str):
    """Extract facemesh features from a video file for with mediapipe

    Args:
        src_video (str): Path to src video
        dst_csv (str): Path to dst feature csv

    Returns:
        df (DataFrame): face features dataframe with features x,y,z as columns
        with the format [feature type]_[number]
    """

    df = _get_features(srv_video)
    ensure_destination_exists(dst_csv)
    df.to_csv(dst_csv, index=False)
    return df


def _get_features(src_video: str):
    mp_face_mesh = mp.solutions.face_mesh
    total_columns = len(FEATURES_TO_KEEP)
    title_list = [f"{l}_{i}" for i in range(total_columns) for l in ["x","y"]]#["x","y","z"]]
    landmark_list = []
    # For video/webcam input:
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(src_video)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
            # continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                data_list = []
                for idx, i in enumerate(face_landmarks.landmark):
                    if idx in FEATURES_TO_KEEP:
                        data_list.append(i.x)
                        data_list.append(i.y)
                        # data_list.append(i.z) # Uncomment to include depth
                landmark_list.append(data_list)
                break
        else:
            landmark_list.append([0 for _ in range(total_columns)])

    face_mesh.close()
    cap.release()

    df = pd.DataFrame(landmark_list, columns=title_list)
    return df

# Just using connected featurepoints from https://google.github.io/mediapipe/solutions/face_mesh.html
FACE_CONNECTIONS = frozenset([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])

FEATURES_TO_KEEP = list(set([i for i,j in FACE_CONNECTIONS]))

if __name__ == "__main__":
    get_facemesh_features("/home/interactionlab/chrisb/modeling-pipeline/data/video/1/short_example.mp4","/home/interactionlab/chrisb/modeling-pipeline/data/test.csv")