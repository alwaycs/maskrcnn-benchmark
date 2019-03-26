import os

from PIL import Image, ImageDraw



def load_annoatation(p):
    # lines = open(p).read()
    with open(p, 'r') as f:
        lines = f.readlines()
    image_paths = []
    data_bboxes = []
    for line in lines:
        bboxes = []
        line = line.split(' ')
        image_path = line[0]
        image_paths.append(image_path)
        # from ipdb import set_trace
        # set_trace()

        # if image_path == 'UCSD/train_data/vidd1_33_009_f019.jpg':
        #   from ipdb import set_trace
        #   set_trace()
        if line[1] == '0':
            # TODO:there are some images not containing heads,
            # now i can not deal with this lines
            continue
        line = map(float, line[1:])

        head_nums = line[0]
        for i in range(int(head_nums)):
            bbox = line[1+i*5:6+i*5]
            bbox = xywh_to_xyxy(bbox[1:])
            bboxes.append(bbox)
        data_bboxes.append(bboxes)
    return image_paths, data_bboxes

def xywh_to_xyxy(bbox):
    # bbox is a list which represent a box(x, y, w, h)

    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def main():
    train_data_dir = "/workspace/csf/data/head_detect/train/yuncong_data/"
    train_label = {'Mall': 'Mall_train.txt',
                    'A': 'Part_A_train.txt',
                    'B': 'Part_B_train.txt',
                    'UCSD': 'UCSD_train.txt',
                    'our': 'our_train.txt'}

    image_paths, data_bboxes = load_annoatation(os.path.join(train_data_dir, train_label['Mall']))
    for (image_path, bboxes) in zip(image_paths, data_bboxes):
        image_path = os.path.join(train_data_dir, image_path)
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for box in bboxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red')
        image.show()
        from ipdb import set_trace
        set_trace()

if __name__ == '__main__':
    p = "/workspace/csf/data/head_detect/yuncong_data/UCSD_train.txt"
    if os.path.isfile(p):
        print('True')
    else:
        print('False')
    main()
