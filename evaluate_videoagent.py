import pickle
import motmetrics as mm
import json
import pandas as pd
pd.set_option('display.max_rows', None)


def compare_captions(
    video_name: str, 
    videoagent_caption_path: str = None, 
    gt_segments_path: str = "annotations/gt_segments.json"
):
    """
    :param video_name: Name of the video used as recorded by IKEA ASM 
        (e.g., Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00)
    :param videoagent_caption_path: The path to the captions created by 
        VideoAgent's temporal memory module.
    :param gt_segments_path: The path to the JSON file contiaining the segments.

    :return: A dictionary that maps intervales (e.g., "0_50") to dictionaries.
        The inner dictionaries have a key "prediction" containing the predicted
        caption, and a key "gt" containing the ground truth action.
    """
    
    # videoagent inferred caption data
    with open(videoagent_caption_path) as f:
        videoagent_captions = json.load(f)

    # ikea asm gt caption data
    with open(gt_segments_path) as f:
        gt_segments_data = json.load(f)

    gt_segs = gt_segments_data["database"][video_name]["annotation"]

    # get the annotations for each clip
    curr_gt_seg = 0
    results = {}
    for interval in videoagent_captions:
        print(f"checking interval {interval}")
        temp = []

        start, end = [int(num) for num in interval.split("_")]
        temp.append(gt_segs[curr_gt_seg]["label"])
        print(f"\tadded segment {gt_segs[curr_gt_seg]['segment']} with action '{gt_segs[curr_gt_seg]['label']}'")

        while gt_segs[curr_gt_seg]["segment"][1] < end:
            curr_gt_seg += 1
            temp.append(gt_segs[curr_gt_seg]["label"])
            print(f"\tadded segment {gt_segs[curr_gt_seg]['segment']} with action '{gt_segs[curr_gt_seg]['label']}'")

        results[interval] = {
            "prediction": videoagent_captions[interval],
            "gt": ", then ".join(temp)
        }
    
    return results


def compare_tracking(annotation_filepath: str, prediction_filepath: str):
    """
    Check tracking metrics
    """

    '''
    pseudo schema of annotation file:
    {
        images: [
            {id: int, filename: str} // extract frame number from filename, id = index + 1, index = id - 1
            // filename like Lack_Coffee_Table/0001_black_floor_01_01_2019_08_14_15_22/dev3/images/000044.jpg
        ],

        annotations: [
            {
                category_id: int, // same as "id" in categories
                image_id: int, // same as "id" in images
                segmentation: [int] // a polygon for the object
                bbox: [int] // a rectangle for the object
                part_id: int // for tracking individual parts
                ...
            }
        ],

        categories: [
            {
                id: int, // id = index + 1
                name: str, // name of the object being tracked
                ...
            }
        ]
    }
    '''
    # the frames with tracking data

    with open(annotation_filepath) as f:
        annotation_data = json.load(f)

    # Convert COCO into MOTChallenge
    images = annotation_data["images"]
    annotations = annotation_data["annotations"]

    # The set of frames that were annotated
    frames_annotated = set()

    # keys are frame numbers. Values are tuples (category_id, bbox[0], ...)
    annotation_motc = {}
    for annotation in annotations:
        # extract info needed for MOTChallenge format
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        part_id = annotation["part_id"]

        # extract frame number from filename
        filename = images[image_id - 1]["file_name"].rsplit("/", 1)[-1]
        frame_num = int(filename[:filename.index(".")])

        # add current entry to motc
        if frame_num not in annotation_motc:
            annotation_motc[frame_num] = [(part_id, *bbox)]
        else:
            annotation_motc[frame_num].append((part_id, *bbox))

        frames_annotated.add(frame_num)

    '''
    pseudo schema for rt-detr tracking (re-id)
    scheme 1:
    [
        {
            part_id (int): [category_name: str, [vertices...]]
        }
    ]

    scheme 2:
    index is part id, value is array of frame numbers in which part 
    appears (index into scheme 1)

    scheme 3:
    {part_id (int): category_name (str)}
    '''
    with open(prediction_filepath, "rb") as f:
        prediction_data = pickle.load(f)

    segmentations = prediction_data[0]
    part_appearances = prediction_data[1]

    # keys are frame numbers. Values are tuples (part_id, bbox[0], ...)
    prediction_motc = {}
    for part in range(len(part_appearances)):
        for frame_num in part_appearances[part]:
            if frame_num not in prediction_motc:
                prediction_motc[frame_num] = [(part, *segmentations[frame_num][part][1])]
            else:
                prediction_motc[frame_num].append((part, *segmentations[frame_num][part][1]))

    # calculate MOTA
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_num in annotation_motc:
        if frame_num not in prediction_motc:
            continue

        objs = annotation_motc[frame_num]
        hyps = prediction_motc[frame_num]

        distance_matrix = mm.distances.iou_matrix(
            [obj[1:] for obj in objs], [hyp[1:] for hyp in hyps]
        )

        acc.update(
            [obj[0] for obj in objs], [hyp[0] for hyp in hyps], distance_matrix
        )

    return acc
        


if __name__ == "__main__":
    print('\033c')
    '''
    result = compare_captions("Lack_Coffee_Table/0001_black_floor_01_01_2019_08_14_15_22", "preprocess/scan_video.avi/captions.json")
    for key, value in result.items():
        print(key, "prediction:\t", value['prediction'])
        print(key, "gt:\t\t", value['gt'])
        print()
    '''
    
    metrics= compare_tracking(
        "annotations/Final_Annotations_Segmentation_Tracking/Lack_Coffee_Table/0001_black_floor_05_02_2019_08_19_16_51/dev3/manual_coco_format_with_part_id.json",
        "preprocess/scan_video.avi/reid.pkl"
    )

    events = metrics.mot_events
    events = events[events["Type"].notnull()]
    print(events[events["Type"] == "MATCH"])

    