import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from utils.augment import *
from utils.utils import plot_gt
from utils.bbox import quad_2_rbox, constraint_theta

class GaoFenShipDataset(data.Dataset):
    def __init__(self,
                 dataset= None,  # 训练的txt文件（内含所有文件名，不用绝对路径）
                 augment = False,
                 level = 1,
                 ):
        self.image_set_path = dataset
        if dataset is not None:
            self.image_list = self._load_image_names()  # 返回所有文件名的list，如['img_145.jpg',  'img_259.jpg', ...]
        self.level = level
        self.classes = ('__background__', 'ship') 
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))    # 压缩成dict：{'__background__': 0, 'text': 1}
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # index是索引号,每次进8线程同时进行处理（num_worker），self.image_list[index]从trainval.txt的所有img中提取出遍历当前的图片名
        im_path = self.image_list[index]   # 获取当前张单张图像的绝对路径
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # 由当前的文件提取出xml的信息,dict格式内含两对kv：'boxes'-shape(nt,8)和'gt_classes'-shape(1,nt)
        roidb = self._load_annotation(self.image_list[index])
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
        nt = len(roidb['boxes'])
        # gt_boxes的格式为：xyxyac
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        if nt:
            bboxes = roidb['boxes'][gt_inds, :]
            classes = roidb['gt_classes'][gt_inds]
            if self.augment :
                transform = Augment([HorizontalFlip(0.5),
                                    VerticalFlip(0.5),
                                    Affine(degree=0, translate=0.1, scale=0.1, p=0.5),
                                    # Grayscale(0.3,p=0.5),
                                    # # Contrast(0.15, p=0.3),
                                    # Sharpen(0.15, p=0.2),
                                    Noise(0.1, p=0.5),
                                    # Gamma(0.2, p=0.4),
                                    # Blur(1.3, p=0.5),
                                    ],box_mode = 'xywha',)
                im, bboxes = transform(im, bboxes)
            gt_boxes[:, :-1] = bboxes

            mask = mask_valid_boxes(quad_2_rbox(bboxes,'xywha'), return_mask=True)
            bboxes = bboxes[mask]
            gt_boxes = gt_boxes[mask]
            classes = classes[mask]

            for i, bbox in enumerate(bboxes):
                gt_boxes[i, 5] = classes[i]
            gt_boxes = constraint_theta(gt_boxes)
            cx, cy, w, h = [gt_boxes[:, x] for x in range(4)]      
            x1 = cx - 0.5*w
            x2 = cx + 0.5*w
            y1 = cy - 0.5*h
            y2 = cy + 0.5*h
            gt_boxes[:,0] = x1;  gt_boxes[:,1] = y1; gt_boxes[:,2] = x2; gt_boxes[:,3] = y2

            ## test augmentation
            print(im.shape)
            plot_gt(im, gt_boxes[:,:5], im_path, mode = 'xyxya')

        return {'image': im, 'boxes': gt_boxes, 'path': im_path}

    def _load_image_names(self):
        """
        Load the names listed in this dataset's image set file.
        """
        image_set_file = self.image_set_path
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]
        return image_list


    def _load_annotation(self, index):
        root_dir, img_name = os.path.split(index)
        filename = os.path.join(root_dir, img_name[:-5]+'.xml')
        
        boxes, gt_classes = [], []
        with open(filename,'r',encoding='utf-8-sig') as f:
            content = f.read()
            assert '<objects>'  in content, 'Background picture occurred in %s'%filename
            objects = content.split('<object>')
            info = objects.pop(0)
            for obj in objects:
                assert len(obj) != 0, 'No onject found in %s'%filename
                points = obj[obj.find('<points>')+8: obj.find('</points>')].split('<point>')[1:]
                coors = [x.split('<')[0].split(',')  for x in points]
                (x1,y1),(x2,y2),(x3,y3),(x4,y4),*_ = coors
                x1, y1, x2, y2, x3, y3, x4, y4 = [x for x in map(eval, [x1, y1, x2, y2, x3, y3, x4, y4])]
                quad_box = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
                # 统一为角度制
                box = quad_2_rbox(quad_box)
                boxes.append(box)
                label = self.class_mapping(cls_id, self.level)
                gt_classes.append(label)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}

    def class_mapping(self, cls_id, level):
        if level == 1:
            return 1
        if level == 2:
            if cls_id in ['100000005','100000006','100000012','100000013','100000016','10000032']:
                cls_id =  '100000002'
            if cls_id in ['100000007','100000008','100000009','100000010','100000011','10000015',
                          '10000017','10000019','10000028','10000029']:
                cls_id =  '100000003'
            if cls_id in ['100000018','100000020','100000024','100000025','100000026','10000030']:
                cls_id = '100000004'
            class_ID = ['bg', '100000001', '100000002', '100000003', '100000004']
            return class_ID.index(cls_id)
        if level == 3:
            if cls_id in ['1000000012', '1000000013', '1000000032',]:
                cls_id = '100000002'
            if cls_id in ['100000017', '100000028']:
                cls_id = '100000003'
            if cls_id in ['100000024', '100000026']:
                cls_id = '100000004'
            class_ID = ['bg', '100000001', '100000002', '100000003', '100000004', '100000005', 
                        '100000006', '100000007', '100000008', '100000009', '1000000010',
                        '1000000011', '100000015','1000000016', '100000018', '100000019', 
                        '100000020', '100000025' , '100000029', '100000030'] 
            return class_ID.index(cls_id)

    def return_class(self, id):
        id = int(id)
        return self.classes[id]

            


if __name__ == '__main__':
    pass
