import argparse
from PIL import Image
from yolo import YOLO
from train import Train


def get_parser():
    parser = argparse.ArgumentParser()

    # preditct
    parser.add_argument("--predict", default=False, help="predict the picture")
    parser.add_argument("--img_path", type=str, default="img\street.jpg", help="input image path")
    parser.add_argument("--crop", default=False, help="Is crop the picture")
    parser.add_argument("--count", default=False, help="count")

    # train
    parser.add_argument("--train", default=False, help="train the model")
    parser.add_argument("--classes_path", type=str, help="input classes path")
    parser.add_argument("--anchors_path", type=str, help="input anchors path")
    parser.add_argument("--model_path", type=str, help="input model path")
    parser.add_argument("--phi", type=str, default='s', help="input phi")
    parser.add_argument("--train_annotation_path", type=str, help="input train_annotation path")
    parser.add_argument("--val_annotation_path", type=str, help="input val_annotation path")
    parser.add_argument("--train_epoch", type=int, default='300', help="train epoch")

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    
    if args.predict:
        image = Image.open(args.img_path)
        crop = args.crop
        count = args.count

        yolo = YOLO()

        r_image = yolo(image, crop=crop, count=count)
        r_image.show()


    elif args.train:
        classes_path = args.classes_path
        anchors_path = args.anchors_path
        model_path = args.model_path
        phi = args.phi
        train_annotation_path = args.train_annotation_path
        val_annotation_path = args.val_annotation_path
        train_epoch = args.train_epoch
        
        Train(classes_path, anchors_path, model_path, train_annotation_path, val_annotation_path, phi, train_epoch)


    else:
        print("你没有提供有效的参数!")
        print("请在命令行传入如下参数: ")
        print("python main.py --train True --classes_path model_data/voc_classes.txt --anchors_path model_data/yolo_anchors.txt --model_path model_data/yolov5_s.pth --train_annotation_path 2007_train.txt --val_annotation_path 2007_val.txt")