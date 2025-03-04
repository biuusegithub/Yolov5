import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

'''
VOC 数据集标签划分处理
'''

# 用于在.xml文件中找到真实框坐标信息并写入 list_file
def convert_annotation(VOCdevkit_path, year, image_id, list_file, classes, nums):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()


def voc_annotation(annotation_mode, classes_path, trainval_percent, train_percent, VOCdevkit_path, classes, nums, VOCdevkit_sets=[('2007', 'train'), ('2007', 'val')]):
    random.seed(0)
    photo_nums  = np.zeros(len(VOCdevkit_sets))

    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格")


    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)  
        list = range(num)  
        tv = int(num*trainval_percent)  
        tr = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")


    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0

        for year, image_set in VOCdevkit_sets:

            # 这里一前一后的占字符 %s 分别用于放入 year、image_set
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')

            for image_id in image_ids:
                # abspath() 返回绝对路径
                # 把每张图的绝对路径写入 list_file
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                # 把gt坐标信息写入 list_file
                convert_annotation(VOCdevkit_path, year, image_id, list_file, classes, nums)

                list_file.write('\n')

            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        
        print("2007_train.txt and 2007_val.txt finish.")

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500, 属于较小的数据量, 请注意设置较大的训练Epoch, 以满足足够的梯度下降次数。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标, 请注意修改classes_path对应自己的数据集, 并且保证标签名字正确, 否则训练将会没有任何效果！")



if __name__ == '__main__':
    # annotation_mode为 0 代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
    # annotation_mode为 1 代表获得VOCdevkit/VOC2007/ImageSets里面的txt
    # annotation_mode为 2 代表获得训练用的2007_train.txt、2007_val.txt
    annotation_mode = 2

    classes_path = 'model_data/voc_classes.txt'

    # 训练集+验证集与测试集、训练集+验证集, 默认按照 9:1的划分
    trainval_percent = 0.9
    train_percent = 0.9

    # 指向VOC数据集所在的文件夹
    VOCdevkit_path = 'VOCdevkit'

    VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]

    classes, _ = get_classes(classes_path)
    # 统计目标数量
    nums = np.zeros(len(classes))

    voc_annotation(annotation_mode, classes_path, trainval_percent, train_percent, VOCdevkit_path, classes, nums)
