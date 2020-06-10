import pandas as pd
import json
import re
import os
import numpy as np
from sympy import Point3D, Line3D, Plane
import cv2 as cv
import seaborn as sns
import matplotlib.pyplot as plt


# App

def app(filepath, queue):
    queue.put("START_PROCESSING")

    # Output paths

    basePath = './Vuelosophy_IO/'
    path_h264 = ''
    path_mp4 = basePath + 'output_MP4/'
    path_json = basePath + 'output_JSON/'
    path_pkl = basePath + 'output_PKL/'

    # Output files

    # filename = os.listdir('./Vuelosophy_IO/input_H264/')[0].rstrip(".h264")
    filename = filepath.split("/")
    filename = filename[len(filename) - 1].rstrip(".h264")

    file_h264 = filepath
    file_mp4 = filename + '.mp4'
    file_json = filename + '.json'
    file_pkl = filename + '.pkl'

    queue.put("FILENAME:%s" % filename)

    # Create JSON file from H264
    queue.put("STEP: Creating JSON file")
    h264_to_json(path_h264, file_h264, path_json, file_json)

    # Create and split dataframe from json
    queue.put("STEP: Split Dataframe")
    df_header, df_body = header_body_split(json_to_pandas_df(path_json, file_json))

    # Pickle both dataframes
    queue.put("STEP: Pickle dataframe (header)")
    pd.to_pickle(df_header, path_pkl + 'pickled_df_header_' + file_pkl)
    # pd.to_pickle(df_body, path_pkl + 'pickled_df_body_' + file_pkl)

    # Create Frame object
    Frame.width = df_header['frame_width']
    Frame.height = df_header['frame_height']
    Frame.rate = df_header['frame_rate']
    Frame.total = df_header['frame_total']
    Frame.number = df_body['frame_number'].iloc[:]

    # Create Unity object
    Unity.width = 13.333
    Unity.height = 10
    Unity.distance = 12.678
    frame_ratio = Frame.height / Unity.height

    # Preprocessing
    queue.put("STEP: Preprocessing")
    if os.path.exists(path_pkl + 'pickled_df_body_preprocessed' + file_pkl):
        df_body = pd.read_pickle(path_pkl + 'pickled_df_body_preprocessed' + file_pkl)
    else:
        df_body = preprocess(df_body, Unity, Frame, path_pkl, file_pkl)

    Video.window_name = 'eye tracking (Vuelosophy)'
    Video.nr_of_frames = Frame.total - 2
    Video.input_file = path_h264 + file_h264
    Video.output_file = path_mp4 + file_mp4

    # Processing video
    queue.put("STEP:Creating video")
    print(path_mp4 + filename + file_mp4)
    if os.path.exists(path_pkl + 'pickled_df_body_vid' + file_pkl) and os.path.exists(path_mp4 + file_mp4):
        df_body = pd.read_pickle(path_pkl + 'pickled_df_body_vid' + file_pkl)
    else:
        df_body = create_video(df_body, Video, Frame)
        pd.to_pickle(df_body, path_pkl + 'pickled_df_body_vid' + file_pkl)
    print("done vid processing")
    # Countplot
    # sns.countplot(data=df_body, x="object")

    # Timelineplot
    # gantplot(df_body, filename)

    queue.put("END_PROCESSING")


class Frame:
    width = 0
    height = 0
    rate = 0.0
    total = 0
    number = 0
    timestamp = 0.0  # seconds


class Unity:
    width = 0.0
    height = 0
    distance = 0.0


class Video:
    window_name = ''
    input_file = ''
    output_file = ''
    nr_of_frames = 0


def h264_to_json(path_h264, file_h264, path_json, file_json):
    # Open the file in Read Binary mode and read it
    with open(path_h264 + file_h264, 'rb') as f:
        data = f.read()

    # Use regular expressions to extract the coordinates
    result = re.findall(b'(?<=\xff\xff\xff\xff\xff)\{.+?(?=\x00\x00\x00\x01)', data)
    result = [str(part)[2:-1] for part in result]

    # Create JSON
    regular_json = '[' + ',\n'.join(result) + ']'

    # Prettify JSON
    pretty_json = json.loads(regular_json)

    # Create new JSON-file
    with open(path_json + file_json, 'w') as f:
        # f.write(regularjson)#compact (on one line)
        f.write(json.dumps(pretty_json, indent=4))


def json_to_pandas_df(path_json, file_json):
    # Open JSON-file
    with open(path_json + file_json) as f:
        data = json.load(f)

    # Normalize data
    df_all = pd.json_normalize(data)

    # Return the whole dataframe
    return df_all


def header_body_split(df_all):
    # Locate header, rename it and save it

    df_header = df_all.loc[[0], ['width', 'height', 'frame_rate', 'frame_total']]
    df_header.rename(columns={'width': 'frame_width', 'height': 'frame_height'}, inplace=True)

    df_header['frame_width'] = df_header['frame_width'].astype(int)
    df_header['frame_height'] = df_header['frame_height'].astype(int)
    df_header['frame_total'] = df_header['frame_total'].astype(int)

    # Locate body, rename it and save it
    df_body = df_all[['num', 'ft.x', 'ft.y', 'ft.z']]
    df_body = df_body.iloc[1:]
    df_body.rename(columns={'num': 'frame_number', 'ft.x': 'norm_eye_3d_x', 'ft.y': 'norm_eye_3d_y', 'ft.z': 'norm_eye_3d_z'},
                   inplace=True)

    df_body['frame_number'] = df_body['frame_number'].astype(int)

    # Return the header converted to dictionary
    return df_header.iloc[0].to_dict(), df_body


def preprocess(df_body, unity, frame, path_pkl, file_pkl):
    # Convert 3D coords to 2D coords
    df_body = coord_3d_to_2d(df_body, unity, frame)

    # Calculate fixations
    df_body = get_fixations(df_body)

    # Pickle dataframe body
    pd.to_pickle(df_body, path_pkl + 'pickled_df_body_preprocessed' + file_pkl)

    return df_body


def coord_3d_to_2d(df_body, unity, frame):
    # Variables
    unity_width_half = unity.width / 2
    unity_height_half = unity.height / 2
    unity_distance = unity.distance
    list_px_2d_x = []
    list_px_2d_y = []

    # Create a Plane
    plane = Plane(Point3D(-unity_width_half, unity_height_half, -unity_distance),
                  Point3D(-unity_width_half, -unity_height_half, -unity_distance),
                  Point3D(unity_width_half, unity_height_half, -unity_distance))

    # Calculate the coords for every item in the dataframe
    for ind in df_body.index:
        dummy_px_2d_x, dummy_px_2d_y = calculate_coord(plane, unity, frame, df_body['norm_eye_3d_x'][ind], df_body['norm_eye_3d_y'][ind], df_body['norm_eye_3d_z'][ind])
        list_px_2d_x.append(int(dummy_px_2d_x))
        list_px_2d_y.append(int(dummy_px_2d_y))

        # Print progress
        print("%s / %s" % (ind, len(df_body.index)), end="\r")

    df_body.loc[:, 'px_eye_2d_x'] = list_px_2d_x
    df_body.loc[:, 'px_eye_2d_y'] = list_px_2d_y

    return df_body


def calculate_coord(plane, Class_unity, Class_frame, norm_eye_3d_x, norm_eye_3d_y, norm_eye_3d_z):
    unity_width_half = Class_unity.width / 2
    unity_height_half = Class_unity.height / 2
    frame_width = Class_frame.width
    frame_height = Class_frame.height
    line = Line3D(Point3D(0, 0, 0),
                  Point3D(norm_eye_3d_x, norm_eye_3d_y, norm_eye_3d_z))
    intersection = plane.intersection(line)[0]
    px_eye_2d_x = map_value(intersection.x, -unity_width_half, unity_width_half, 0, frame_width)
    px_eye_2d_y = map_value(intersection.y, -unity_height_half, unity_height_half, frame_height, 0)
    return px_eye_2d_x, px_eye_2d_y


def map_value(value, left_min, left_max, right_min, right_max):
    left_span = left_max - left_min
    right_span = right_max - right_min
    value_scaled = (value - left_min) / left_span  # convert the left range into a 0-1 range
    return np.nan_to_num(right_min + (value_scaled * right_span))  # convert the 0-1 range into a value in the right range


def get_fixations(df_body):
    # Position the data
    df_body['px_eye_2d_x2'] = df_body['px_eye_2d_x'].shift(-1)
    df_body['px_eye_2d_y2'] = df_body['px_eye_2d_y'].shift(-1)

    # Delete last row (empty)
    df_body = df_body[:-1]

    # Calculate the eye distance
    df_body['eye_dist'] = ((df_body['px_eye_2d_x2'] - df_body['px_eye_2d_x']) ** 2) + ((df_body['px_eye_2d_y2'] - df_body['px_eye_2d_y2']) ** 2)
    q1 = np.percentile(df_body['eye_dist'], 25)
    q3 = np.percentile(df_body['eye_dist'], 75)

    # Calculate the threshold
    threshhold = q3 + (1.5 * (q3 - q1))

    # Check if it is a fixation or not
    df_body['fixation'] = df_body.apply(lambda x: row_fixation(x, threshhold), axis=1)

    return df_body


def row_fixation(row, threshhold):
    if row.eye_dist <= threshhold:
        return 1
    else:
        return 0


def create_video(df_body, Video, Frame):
    # Variables
    current_frame = 0

    # cv.namedWindow(Video.window_name)

    # Get the video
    cap = cv.VideoCapture(Video.input_file)

    # Create a video writer
    codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v')  # MP4
    writer = cv.VideoWriter(Video.output_file, codec, int(Frame.rate), (int(Frame.width), int(Frame.height)), True)

    # Create a object ID
    df_body["objectId"] = np.nan
    numpy_array = df_body[["px_eye_2d_x", "px_eye_2d_y", "fixation"]].to_numpy()

    # Prepare object detection
    classes = ["Jam", "Knife", "Bread", "Choco"]
    net = cv.dnn.readNet("Weights/TinyWeightsZarinV4.weights", "Configs/TinyConfigZarin.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True and current_frame < Video.nr_of_frames:

            # Object detection
            height, width, channels = frame.shape
            blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations in the frame
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # Object detected
                        # print("%s: %s " % (class_id, confidence) + '%')
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv.putText(frame, label, (x, y + 30), cv.FONT_HERSHEY_PLAIN, 2, color, 2)

                    # object in vision
                    if numpy_array[current_frame][0] in range(x, x + w) and numpy_array[current_frame][1] in range(y, y + h):
                        if numpy_array[current_frame][2] == 1:
                            print("looking at: %s, frame: %s" % (label, current_frame))
                            df_body.at[current_frame + 1, "objectId"] = class_ids[i]

            # Show the eye focus
            if numpy_array[current_frame][2] == 1:
                # Fixation
                cv.circle(frame, (numpy_array[current_frame][0], numpy_array[current_frame][1]), 30, (255, 255, 255), 3)

            if numpy_array[current_frame][2] == 0:
                # Saccade
                cv.circle(frame, (numpy_array[current_frame][0], numpy_array[current_frame][1]), 30, (0, 0, 255), 3)

            # cv.imshow(Video.window_name, frame)
            # if cv.waitKey(1) & 0xFF == ord('q'):  # press Q on keyboard to exit
            #     break

            writer.write(frame)

            current_frame = current_frame + 1

        else:
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()
    df_body['object'] = df_body.apply(lambda x: name_object(x), axis=1)
    return df_body


def name_object(row):
    classes = ["Jam", "Knife", "Bread", "Choco"]
    if row['objectId'] in range(0, len(classes)):
        return classes[int(row['objectId'])]


def gantplot(df_body, filename):
    usefull = df_body.dropna()
    viewed_classes = usefull['object'].unique()
    graph_data = {}
    for classname in viewed_classes:
        graph_data[classname] = usefull[usefull['object'] == classname]

    fig, gnt = plt.subplots()

    gnt.set_yticks(np.arange(10, len(viewed_classes) * 10 + 1, 10))
    gnt.set_yticklabels(viewed_classes)
    gnt.set_xlabel('tijdverloop')

    gnt.set_yticklabels(viewed_classes)
    gnt.grid(True)

    colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:orange']
    for key, val in graph_data.items():
        output = [(row['frame_number'], 1) for index, row in val.iterrows()]
        nr = list(viewed_classes).index(key)
        gnt.broken_barh(output, (nr * 10 + 5, 10), facecolors=colors[nr])
    # plt.show()
    return gnt
    # plt.savefig(filename + "_gantplot.png")
