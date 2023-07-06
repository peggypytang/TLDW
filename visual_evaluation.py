import os
import time
import numpy
import argparse
from os.path import join, exists
import glob
import cv2
import numpy as np


def main(args):
    dec_pos_dir = join(args.decode_dir, args.decode_pos_folder)
    dec_dir = join(args.decode_dir, args.decode_folder)
    split = 'test'
    ref_dir = join(args.ref_dir, split)
    assert exists(ref_dir)


    dec_pos_files = glob.glob(join(dec_pos_dir, "*.dec"))
    file_id = [os.path.split(x)[1].replace('.dec', '') for x in dec_pos_files]

    decode_pos = []
    decode_pic = []
    ref_pic = []
    count_frame = []
    thumbnail = []

    for i in file_id:
        with open(join(dec_pos_dir, '{}.dec'.format(i)),'r') as f:
            decode_pos.append(f.read())

        decode_pic.append(cv2.imread(join(dec_dir, '{}.png'.format(i))))

        
        ref_pic.append(cv2.imread(join(ref_dir, '{}.png'.format(i))))

        
        vidcap = cv2.VideoCapture(join(ref_dir, '{}.mp4'.format(i)))
        success,image = vidcap.read()
        
        i = 0
        count = 0
        while success:
            if i % 120:
                count +=1 
            success, image = vidcap.read()
            
            if count == int(decode_pos[-1].replace("tensor([", "").replace("], device='cuda:0')", "")):
                thumbnail.append(image)

            i += 1
        count_frame.append(count)

    output, output_list = analysis_accuracy(thumbnail, ref_pic)
    metric = 'accuracy'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('accuracy: ' + str(output) + '\n' + '\n'.join([str(i) for i in output_list]))
    print(metric, output)

    output, output_list, decode_object_list, ref_object_list = analysis_iou(thumbnail, ref_pic)
    metric = 'iou'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('iou: ' + str(output) + '\n' + '\n'.join([str(i) for i in output_list]))

    print(metric, output)

    with open(join(args.decode_dir, '{}.txt'.format("decode_object_list")), 'w') as f:
        f.write(str('\n'.join(decode_object_list)))
    with open(join(args.decode_dir, '{}.txt'.format("ref_object_list")), 'w') as f:
        f.write(str('\n'.join(ref_object_list)))

    with open(join(args.decode_dir, '{}.txt'.format("file_id")), 'w') as f:
        f.write(str('\n'.join(file_id)))

        


def calculateEudDistance(i1, i2): #lower-level visual features 
    distance = np.linalg.norm(i1-i2)
    norm_dis = np.linalg.norm(np.ones(( 360,640,3))*255)
    distance = distance / norm_dis
    return distance


def analysis_accuracy(decode_list, ground_list): #DeepQAMVS: Query-Aware Hierarchical Pointer Networks for Multi-Video Summarization
    correct = 0
    distance = []
    for i1, i2 in zip(decode_list, ground_list):
        i2_resize = cv2.resize(i2, (640, 360))
        i1_resize = cv2.resize(i1, (640, 360))
        dis = calculateEudDistance(i1_resize, i2_resize)
        distance.append(dis)
        if dis <= 0.6:
            correct += 1
    acc = correct/len(decode_list)
    return acc, distance

def get_objects(image, net):
    CONF = 0.5
    THRESHOLD = 0.5
    # load our input image and grab its spatial dimensions
    labelsPath = os.path.sep.join([args.yolo_folder, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
    # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONF:
                confidences.append(float(confidence))
                classIDs.append(classID)

    detected_labels = []

    for i in classIDs:
        detected_labels.append(LABELS[i])

    return detected_labels

def get_iou(item1,item2):
    

    intersection = 0
    for item in item1:
        if item in item2:
            intersection+=1
    union = len(item1)+len(item2)-intersection

    if union > 0:
        return intersection*1.0 / union
    else:
        return 1

def analysis_iou(decode_list, ground_list): #Deep Reinforcement Learning for Query-Conditioned Video Summarization
    #higher-level semantic information
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args.yolo_folder, "yolov3.weights"])
    configPath = os.path.sep.join([args.yolo_folder, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    iou_list = []
    i1_object_list = []
    i2_object_list = []
    for i1, i2 in zip(decode_list, ground_list):
        i1_object = get_objects(i1, net)
        i2_object = get_objects(i2, net)
        i1_object_list.append(','.join(i1_object))
        i2_object_list.append(','.join(i2_object))
        iou_list.append(get_iou(i1_object,i2_object))

    return sum(iou_list)/len(iou_list), iou_list, i1_object_list, i2_object_list

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the output files for the RL full models')

    parser.add_argument('--ref_dir', action='store', required=True,
                        help='directory of ref summaries')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--decode_folder', action='store', required=True,
                        help='folder of decoded summaries')
    parser.add_argument('--decode_pos_folder', action='store', required=True,
                        help='folder of decoded summaries')
    parser.add_argument('--yolo_folder', action='store', required=True,
                        help='yolo folder')
    args = parser.parse_args()
    main(args)

