import argparse
import mmcv
import glob
import numpy as np
import os.path as osp
import os
import shutil
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

change_dict = {'成熟单核细胞': '成熟单核细胞', '成熟淋巴细胞': '成熟淋巴细胞', '淋巴细胞': '成熟淋巴细胞', '分裂相': '核分裂相', '成熟浆细胞': '浆细胞', '多核浆细胞': '浆细胞', '双核浆细胞': '浆细胞', '浆细胞': '浆细胞', '巨晚幼红细胞': '巨晚幼红细胞', '巨幼样晚幼': '巨晚幼红细胞', '巨晚幼红细胞伴核畸形': '巨晚幼红细胞', '巨晚幼红细胞伴豪焦小体': '巨晚幼红细胞', '巨原始红细胞': '巨原始红细胞', '巨早幼红细胞': '巨早幼红细胞', '巨中性分叶核粒细胞': '巨中性分叶核粒细胞', '巨中性分叶核幼粒细胞': '巨中性分叶核粒细胞', '巨杆状核粒细胞': '巨中性杆状核粒细胞', '巨幼样杆': '巨中性杆状核粒细胞', '巨中性杆状核粒细胞': '巨中性杆状核粒细胞', '巨幼样杆状核': '巨中性杆状核粒细胞', '巨中性杆状核幼粒细胞': '巨中性杆状核粒细胞', '巨杆状核': '巨中性杆状核粒细胞', '巨晚幼粒细胞': '巨中性晚幼粒细胞', '巨中性晚幼粒细胞': '巨中性晚幼粒细胞', '巨中性中幼粒细胞': '巨中性中幼粒细胞', '巨中幼粒细胞': '巨中性中幼粒细胞', '巨中幼红细胞': '巨中幼红细胞', '粒细胞核畸形': '粒细胞核畸形', '裸核型巨核细': '裸核型巨核细胞', '内皮细胞': '内皮细胞', '嗜碱': '嗜碱性粒细胞', '嗜碱性粒细胞': '嗜碱性粒细胞', '嗜碱晚': '嗜碱性晚幼粒细胞', '嗜酸性粒细胞': '嗜酸性分叶核粒细胞', '嗜酸性分叶核粒细胞': '嗜酸性分叶核粒细胞', '嗜酸性分叶': '嗜酸性分叶核粒细胞', '嗜酸分叶': '嗜酸性分叶核粒细胞', '嗜酸分叶核': '嗜酸性分叶核粒细胞', '嗜酸杆': '嗜酸性杆状核粒细胞', '嗜酸性杆': '嗜酸性杆状核粒细胞', '嗜酸性杆状核粒细胞': '嗜酸性杆状核粒细胞', '晚幼酸': '嗜酸性晚幼粒细胞', '嗜酸晚': '嗜酸性晚幼粒细胞', '嗜酸性晚幼粒': '嗜酸性晚幼粒细胞', '嗜酸性晚幼粒细胞': '嗜酸性晚幼粒细胞', '中幼酸': '嗜酸性中幼粒细胞', '嗜酸中': '嗜酸性中幼粒细胞', '嗜酸性中幼粒细胞': '嗜酸性中幼粒细胞', '嗜酸性中幼粒': '嗜酸性中幼粒细胞', '退化细胞': '退化细胞', '吞噬细胞': '吞噬细胞', '含H-J小体的晚幼红细胞': '晚幼红细胞', '含H-J小体晚幼红细胞': '晚幼红细胞', '晚幼红细胞': '晚幼红细胞', '网状细胞': '网状细胞', '小巨核细胞': '小巨核细胞', '异常早幼粒': '异常早幼粒细胞', '异常早幼粒细胞': '异常早幼粒细胞', '异型淋巴细胞': '异型淋巴细胞', '幼稚单核细胞': '幼稚单核细胞', '幼单核细胞': '幼稚单核细胞', '幼稚浆细胞': '幼稚浆细胞', '幼稚巨核细胞': '幼稚巨核细胞', '幼稚淋巴细胞': '幼稚淋巴细胞', '幼淋巴细胞': '幼稚淋巴细胞', '原单核细胞': '原始单核细胞', '原始单核细胞': '原始单核细胞', '原始红细胞': '原始红细胞', '原始浆细胞': '原始浆细胞', '原始巨核细胞': '原始巨核细胞', '原始粒细胞': '原始粒细胞', '原始淋巴细胞': '原始淋巴细胞', '早幼红细胞': '早幼红细胞', '早幼粒细胞': '早幼粒细胞', '分叶核粒细胞': '中性分叶核粒细胞', '中性分叶核粒细胞': '中性分叶核粒细胞', '中幼分叶核粒细胞': '中性分叶核粒细胞', '分叶核': '中性分叶核粒细胞', '中性杆状核粒细胞': '中性杆状核粒细胞', '杆状核粒细胞': '中性杆状核粒细胞', '中性杆状核细胞': '中性杆状核粒细胞', '中性杆状核幼粒细胞': '中性杆状核粒细胞', '杆状核': '中性杆状核粒细胞', '中性晚幼粒细胞': '中性晚幼粒细胞', '晚幼': '中性晚幼粒细胞', '晚幼粒细胞': '中性晚幼粒细胞', '中幼粒细胞': '中性中幼粒细胞', '中性中幼粒细胞': '中性中幼粒细胞', '中性中幼红细胞': '中幼红细胞', '中幼红细胞': '中幼红细胞', '组织嗜碱细胞': '组织嗜碱细胞'}
all_classes = ['中性中幼粒细胞', '退化细胞', '吞噬细胞', '中性分叶核粒细胞', '原始单核细胞', '内皮细胞', '原始淋巴细胞', '中幼红细胞', '嗜酸性分叶核粒细胞', '幼稚浆细胞', '巨原始红细胞', '巨晚幼红细胞', '异型淋巴细胞', '嗜酸性中幼粒细胞', '巨早幼红细胞', '原始浆细胞', '组织嗜碱细胞', '粒细胞核畸形', '异常早幼粒细胞', '幼稚淋巴细胞', '成熟淋巴细胞', '幼稚单核细胞', '嗜碱性粒细胞', '巨中性分叶核粒细胞', '裸核型巨核细胞', '幼稚巨核细胞', '嗜碱性晚幼粒细胞', '早幼红细胞', '中性杆状核粒细胞', '核分裂相', '巨中幼红细胞', '早幼粒细胞', '巨中性中幼粒细胞', '巨中性晚幼粒细胞', '嗜酸性晚幼粒细胞', '成熟单核细胞', '嗜酸性杆状核粒细胞', '小巨核细胞', '晚幼红细胞', '浆细胞', '原始巨核细胞', '中性晚幼粒细胞', '网状细胞', '原始红细胞', '巨中性杆状核粒细胞', '原始粒细胞']

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--source-dir', default='/home1/yinhaoli/data/cell/new_data', help='the dir of the source data')
    parser.add_argument(
        '--target-dir', default='/home1/yinhaoli/data/cell/20210708/Annotations', help='the dir to save the generated data')
    parser.add_argument(
        '--train-ratio', default=0.7, type=float, help='ratio of the train number')

    args = parser.parse_args()
    return args

def _find_unlabeled_data(data_root):
    img_data_dir = osp.join(data_root, 'JPEGImages')
    ann_data_dir = osp.join(data_root, 'Annotations')
    img_names = os.listdir(img_data_dir)
    img_names = [img_name.split('.')[0] for img_name in img_names]
    ann_names = os.listdir(ann_data_dir)
    ann_names = [ann_name.split('.')[0] for ann_name in ann_names]
    unlabeled_names = []
    for img_name in img_names:
        if img_name not in ann_names:
            unlabeled_names.append(img_name)

    print(unlabeled_names)


def _count_data_info(target_dir):
    annos = glob.glob(osp.join(target_dir, 'Annotations', '*.xml'))
    class_names = []
    bgrs = []
    for anno in annos:
        tree = ET.parse(anno)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_names.append(name)

    np.random.shuffle(annos)
    for i in range(100):
        anno = annos[i]
        tree = ET.parse(anno)
        root = tree.getroot()
        filename = root.find('filename').text
        img = mmcv.imread(osp.join(target_dir, 'JPEGImages', filename))
        img = mmcv.imresize(img, (img.shape[1] // 10, img.shape[0] // 10))
        bgrs.append(img.reshape(-1, 3))
        print(i)
    bgrs = np.concatenate(bgrs, axis=0)
    mean = np.mean(bgrs, axis=0)
    std = np.std(bgrs, axis=0)
    print(mean, std)
    return set(class_names)

def _generate_division(target_path, train_ratio):
    mmcv.mkdir_or_exist(osp.join(target_path, 'ImageSets', 'Main'))
    names = os.listdir(osp.join(target_path, 'Annotations'))
    np.random.shuffle(names)
    train_names = names[:int(len(names) * train_ratio)]
    val_names = names[int(len(names) * train_ratio):]
    with open(osp.join(target_path, 'ImageSets', 'Main', 'trainval.txt'), 'w') as f:
        for name in names:
            f.write(name.split('.')[0] + '\n')
    with open(osp.join(target_path, 'ImageSets', 'Main', 'train.txt'), 'w') as f:
        for train_name in train_names:
            f.write(train_name.split('.')[0] + '\n')
    with open(osp.join(target_path, 'ImageSets', 'Main', 'val.txt'), 'w') as f:
        for val_name in val_names:
            f.write(val_name.split('.')[0] + '\n')





def _make_xml(target_path, obj_data, image_name):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'cell'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_path = SubElement(node_root, 'path')
    node_path.text = osp.join(target_path, 'JPEGImages', image_name)

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(obj_data))

    img = mmcv.imread(osp.join(target_path, 'JPEGImages', image_name))
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img.shape[1])

    node_height = SubElement(node_size, 'height')
    node_height.text = str(img.shape[0])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for obj_instance in obj_data:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = obj_instance[-1]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(obj_instance[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(obj_instance[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(obj_instance[0]) + int(obj_instance[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(obj_instance[1]) + int(obj_instance[3]))


    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    #print xml 打印查看结果
    return dom

def _copy_data(source_dir, target_dir):
    root_folders = os.listdir(source_dir)
    mmcv.mkdir_or_exist(osp.join(target_dir, 'JPEGImages'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'Annotations'))
    for root_folder in root_folders:
        names = [osp.split(path)[-1].split('.')[0] for path in glob.glob(osp.join(source_dir, root_folder, '*.xml'))]
        for name in names:
            try:
                shutil.copyfile(osp.join(source_dir, root_folder, name+'.jpg'), osp.join(target_dir, 'JPEGImages', name+'.jpg'))
                shutil.copyfile(osp.join(source_dir, root_folder, name+'.xml'), osp.join(target_dir, 'Annotations', name+'.xml'))
            except:
                print(name)

    # annotations = []
    # for root_folder in root_folders:
    #     data_folders = glob.glob(osp.join(root_folder, '*'))
    #     for data_folder in data_folders:
    #         images = glob.glob(osp.join(data_folder, '*.jpg'))
    #         with open(osp.join(data_folder, 'type.txt')) as f:
    #             annotations += f.readlines()
    #         for image in images:
    #             filename = osp.split(image)[-1]
    #             shutil.copyfile(image, osp.join(target_dir, 'JPEGImages', filename))
    # return annotations

def _get_class_names(annotation_path):
    xml_names = os.listdir(annotation_path)
    class_names = []
    for xml_name in xml_names:
        tree = ET.parse(osp.join(annotation_path, xml_name))
        objects = tree.findall('object')
        for object in objects:
            class_names.append(object.find('name').text)

    print(set(class_names))

def _count_class(annotation_path):
    xml_names = os.listdir(annotation_path)
    class_names = {}
    other_names = {}
    for xml_name in xml_names:
        tree = ET.parse(osp.join(annotation_path, xml_name))
        objects = tree.findall('object')
        for object in objects:
            class_name = object.find('name').text
            if change_dict.get(class_name) != None:
                change_name = change_dict[class_name]
                if class_names.get(change_name) == None:
                    class_names[change_name] = 1
                else:
                    class_names[change_name] += 1
            else:
                if other_names.get(class_name) == None:
                    other_names[class_name] = 1
                else:
                    other_names[class_name] += 1

    class_list = sorted(class_names.items(), key=lambda x: x[1])
    other_list = sorted(other_names.items(), key=lambda x: x[1])

    for class_name, value in class_list:
        print('{} {}'.format(class_name, value))
        # print('{}'.format(value))
    print(len(class_list))

    for class_name, value in other_list:
        print('{} {}'.format(class_name, value))
        # print('{}'.format(class_name))
    print(len(other_list))

def _generate_ann(target_dir, annotations):
    annotations_dict = dict()
    mmcv.mkdir_or_exist(osp.join(target_dir, 'Annotations'))
    for annotation in annotations:
        name, x, y, height, width, class_name = annotation.split(',')
        if name not in annotations_dict.keys():
            annotations_dict[name] = [[x, y, height, width, class_name[:-1]]]
        else:
            annotations_dict[name].append([x, y, height, width, class_name[:-1]])

    for i, filename in enumerate(annotations_dict.keys()):
        if filename == '010008610797F9F5026D8B245A023F72AFBA0F76C702.jpg' or not osp.exists(osp.join(target_dir, 'JPEGImages', filename)) or filename == '010009AA7D8185BC63EA5FFBAED2D916A8DE1DC27602.jpg' or filename == '0100098862143E8B22ED715C1305DF63124F68439402.jpg':
            continue
        dom = _make_xml(target_dir, annotations_dict[filename], filename)
        with open(osp.join(target_dir, 'Annotations', filename.split('.')[0] + '.xml'), 'wb') as f:
            f.write(dom.toprettyxml(encoding='utf-8'))
        print(f'{filename} finished')

def _change_names(target_dir, change_dict):
    annos = glob.glob(osp.join(target_dir, 'Annotations', '*.xml'))
    for anno in annos:
        tree = ET.parse(anno)
        objects = tree.findall('object')
        for object in objects:
            object.find('name').text = change_dict[object.find('name').text]
        tree.write(anno)
        print(anno)


def _divide_dataset(source_dir, target_dir, train_ratio):
    names = os.listdir(osp.join(source_dir, 'Annotations'))
    np.random.shuffle(names)
    train_names = names[:int(len(names) * train_ratio)]
    valid_names = names[int(len(names) * train_ratio):]
    with open(osp.join(target_dir, 'train.txt'), 'w') as f:
        for train_name in train_names:
            name = train_name.split('.')[0]
            if not osp.exists(osp.join(source_dir, 'JPEGImages', f'{name}.jpg')):
                continue
            f.write(f'{name}\n')

    with open(osp.join(target_dir, 'valid.txt'), 'w') as f:
        for valid_name in valid_names:
            name = valid_name.split('.')[0]
            if not osp.exists(osp.join(source_dir, 'JPEGImages', f'{name}.jpg')):
                continue
            f.write(f'{name}\n')




def main():
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    train_ratio = args.train_ratio
    # _change_names(source_dir, change_dict)
    # _copy_data(source_dir, target_dir)
    # _get_class_names(osp.join(args.target_dir, 'Annotations'))
    # annotations = _copy_data(source_dir, target_dir)
    # _generate_ann(target_dir, annotations)
    # _generate_division(target_dir, train_ratio)
    # names = _count_data_info(target_dir)
    # print(names)
    # _divide_dataset(source_dir, target_dir, train_ratio)
    # _get_class_names(osp.join(source_dir, 'Annotations'))
    # _find_unlabeled_data(target_dir)
    _count_class(target_dir)
if __name__ == '__main__':
    main()
