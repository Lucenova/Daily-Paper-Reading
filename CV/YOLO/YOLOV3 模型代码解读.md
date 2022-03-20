#### YOLOV3 模型代码解读

##### 1.VOC数据制作

对于VOC数据集，其中分为train dataset和test dataset，那么对应的train dataset中对应的ground truth是以annotations.xml形式给出的。那么首要的任务就是生成一个txt文件，文件的以条作为基本内容：

```txt
../data/VOC/VOCtrainval-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg 263,211,324,339,8 165,264,253,372,8 241,194,295,299,8
图片地址 目标1的坐标和分类 目标2的坐标和分类
```

##### 2.模型训练

- 模型训练的**Dataloader**定义：

  在初始化Dataloader的时候读取第一步生成的VOC数据对应的annotation，即为一个数组，然后在dataloader中的**getitem**中根据传入的index，从annotations读取对应的图片和目标检测框。

  读取图片的时候进行了数据增强：

  - ```python
    class RandomHorizontalFilp(object):
        def __init__(self, p=0.5):
            self.p = p
    
        def __call__(self, img, bboxes):
            # 按照一定的概率将图片翻转
            if random.random() < self.p:
                _, w_img, _ = img.shape
                # 将图片翻转（横向翻转）
                img = img[:, ::-1, :]
                # 图片翻转之后，对应的目标检测框也需要翻转
                bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
            return img, bboxes
    ```

  - ```python
    class RandomCrop(object):
        def __init__(self, p=0.5):
            self.p = p
    
        def __call__(self, img, bboxes):
            if random.random() < self.p:
                h_img, w_img, _ = img.shape
    			# 找到目标框中坐标中的最小的x,y坐标和最大的坐标然后组合成最大的box
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
                
                max_l_trans = max_bbox[0]
                max_u_trans = max_bbox[1]
                max_r_trans = w_img - max_bbox[2]
                max_d_trans = h_img - max_bbox[3]
    			# 裁剪的时候不要取max_bbox和image之间的范围
                crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
                crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
                crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
                crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))
    
                img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
    			# 修改裁剪box的坐标
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
            return img, bboxes
    ```

  - ```python
    class RandomAffine(object):
        def __init__(self, p=0.5):
            self.p = p
    
        def __call__(self, img, bboxes):
            if random.random() < self.p:
                h_img, w_img, _ = img.shape
                # 得到可以包含所有bbox的最大bbox
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
                max_l_trans = max_bbox[0]
                max_u_trans = max_bbox[1]
                max_r_trans = w_img - max_bbox[2]
                max_d_trans = h_img - max_bbox[3]
    
                tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
                ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
    
                M = np.array([[1, 0, tx], [0, 1, ty]])
                # 按照矩阵M对于图片进行变换，类似于矩阵乘法
                img = cv2.warpAffine(img, M, (w_img, h_img))
    
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
            return img, bboxes
    ```

  读取完图片之后，又随机读取了另外一张图片（随机读取图片的时候也进行了相同的augment）进行mix。

  ```python
  class Mixup(object):
      def __init__(self, p=0.5):
          self.p = p
  	
      def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
          if random.random() > self.p:
              lam = np.random.beta(1.5, 1.5)
              img = lam * img_org + (1 - lam) * img_mix
              # 将原图的boxes中每一个添上lam这个参数
              bboxes_org = np.concatenate(
                  [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
              # 将随机选择boxes中每一个添加1-lam
              bboxes_mix = np.concatenate(
                  [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
              bboxes = np.concatenate([bboxes_org, bboxes_mix])
  
          else:
              img = img_org
              bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
  
          return img, bboxes
  ```

  当进行了mix之后，就进入到最关键的一步**create_label**

  ```python
      def __creat_label(self, bboxes):
  		# 读取预先定义好的不同尺度下的anchor和strides
          anchors = np.array(cfg.MODEL["ANCHORS"])
          strides = np.array(cfg.MODEL["STRIDES"])
          train_output_size = self.img_size / strides
          anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]
  		
          # 因为有3个不同的scale，对应不同的scale均生成label，每个scale下label的形状是
          # train_output_size * train_output_size * 3 * 26
          label = [
              np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6 + self.num_classes))
              for i in range(3)]
          # 将不同scale下label中的最后26维向量中的代表置信度的第5维向量数值置为1
          for i in range(3):
              label[i][..., 5] = 1.0
  		# 生成一个矩阵用来储存boxes的x,y,w,h
          bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]  # Darknet the max_num is 30
          # 用来记录3个尺度下的目标框的数目
          bbox_count = np.zeros((3,))
  		
    		# 遍历图片中的目标框
          for bbox in bboxes:
              bbox_coor = bbox[:4]
              bbox_class_ind = int(bbox[4])
              bbox_mix = bbox[5]
  			
              # onehot
              one_hot = np.zeros(self.num_classes, dtype=np.float32)
              one_hot[bbox_class_ind] = 1.0
              one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)
  
              # convert "xyxy" to "xywh"
              bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                          bbox_coor[2:] - bbox_coor[:2]], axis=-1)
              # print("bbox_xywh: ", bbox_xywh)
  			# 将box转换为不同scale下的坐标
              bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
  
              iou = []
              exist_positive = False
              # 因为是3个scale
              for i in range(3):
                  # 对于一个scale，每个grid中会有anchors_per_scale个anchor
                  anchors_xywh = np.zeros((anchors_per_scale, 4))
                  # 将当前的目标box的坐标x,y作为anchor的中心点坐标
                  anchors_xywh[:, 0:2] = np.floor(
                      bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5 
                  # 0.5 for compensation
                  # 将当前scale的anchor赋给anchors_xywh
                  anchors_xywh[:, 2:4] = anchors[i]
  				# 计算anchor和目标box的IOU的值
                  iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                  
                  iou.append(iou_scale)
                  iou_mask = iou_scale > 0.3
  				# 如果当前scale下的对应的3个anchor同目标的box的IOU大于0.3。就相当于
                  # 这个anchor负责该目标的box
                  if np.any(iou_mask):
                      # 将scaled box的x,y坐标转化为整数就相当于grid对应的index
                      xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
  						
                      # 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                      # 这种情况应该是两个目标中心点挨着很近
                      # 其实就是相当于当前的目标的box将之前的box对应的anchor覆盖了
                      label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                      # 这里给IOU大于某个阈值的anchor赋值label为1，作为训练的正样本
                      label[i][yind, xind, iou_mask, 4:5] = 1.0
                      label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                      label[i][yind, xind, iou_mask, 6:] = one_hot_smooth
  					# 这里假设一个图片目标的数目最多是150，超过150就按照取余算
                      bbox_ind = int(bbox_count[i] % 150)
                      # 将第i个scale下的对应的第bbox_ind目标的x,y,w,h记录下来
                      bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                      bbox_count[i] += 1
  
                      exist_positive = True
  			# 假如3个scale下均没有同目标框的IOU大于0.3的anchor
              # 就找到记录的之前iou中最大的作为对应的anchor
              if not exist_positive:
                  best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                  best_detect = int(best_anchor_ind / anchors_per_scale)
                  best_anchor = int(best_anchor_ind % anchors_per_scale)
  
                  xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
  
                  label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                  label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                  label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                  label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth
  
                  bbox_ind = int(bbox_count[best_detect] % 150)
                  bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                  bbox_count[best_detect] += 1
  		
          # 最终生成了不同scale下的哪个anchor对应的x，y, w，h
          label_sbbox, label_mbbox, label_lbbox = label
          # 不同尺度下的box的x,y,w,h
          sbboxes, mbboxes, lbboxes = bboxes_xywh
  
          return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
  
  ```

  

- ##### Yolov3 model流程

  假设输入的size是**3\*3\*416*416**

  经过backbone层对应会有3个输出：

  - x_s：3\*256\*52*52
  - x_m:  3*512\*26\*26

  - x_l: 3*1024\*13\*13

  然后将这3个输出输入到fpn层，对应3个输出，将这3个输出记为**P**：

  - x_s：3\*75\*52*52
  - x_m:  3*75\*26\*26

  - x_l: 3*75\*13\*13

  最后将这个3个输出输入到对应stride生成的head层，在head层中主要是将对应的偏移量x,y的预测值根据计算公式转换为相对应图像的位置坐标，还有就是将w，h对应转换为相对于图片的大小将转换后的结果记为**p_d**。

  将**P**和**p_d**均输入到YOLOV3的损失函数中，计算损失。

- ##### Yolov3损失函数计算

  利用输入到Yolov3损失函数中的**P**获取到对应目标的confidence和所属于class的对应的损失

  利用**p_d**来计算对应每个anchor预测的x,y,w,h的所产生的损失这里用的是giou，这里的giou计算结果是一个数组，对应的大小为batch_size*channels\*img_size\*img_size\*anchor_num\*loss

  最终的loss会将这个loss数组全部加起来。