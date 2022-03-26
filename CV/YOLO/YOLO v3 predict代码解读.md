#### YOLO v3 predict代码解读

Yolo v3的predict的主要代码对应在evaluator.py文件中，对应的predict的代码段如下：

```python
# img是输入的预测图像
# test_shape是模型的输入图像的标准输出448
# valid_scale是（0， inf）
def __predict(self, img, test_shape, valid_scale):
    org_img = np.copy(img)
    org_h, org_w, _ = org_img.shape
	# 将不同大小的图片resize成模型标准的输出448
    # 同时将img的维度增加1维，所以此时img的维度为(1， 3， 448， 448)
    # 如何resize的代码见下文
    img = self.__get_img_tensor(img, test_shape).to(self.device)
    self.model.eval()
    with torch.no_grad():
        # 对应输出p_d的维度是12348 * 25
        # 3个尺度对应的输出分别是（56 * 56 * 3 * 25） （28 * 28 * 3 * 25） （14 * 14 * 3 * 25）
    	_, p_d = self.model(img)
    # pred_bbox对应是12348 * 25
    pred_bbox = p_d.squeeze().cpu().numpy()
    # test_shape是输入的维度448 * 448 
    bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

    return bboxes
```

resize代码：

```python
class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape
		
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
		# 分别计算原始图片宽和高同要resize目标大小的缩放的比例
        # 取宽和高比例最小的作为resize_ratio
        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))
		# 生成一个目标大小的容器，之后会将image_resized放到容器中
        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        # 将image_resized的图片放到容器的中间
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image

```



然后关键的是**__convert_pred**生成pred预测框的部分：

```python
    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        预测框进行过滤，去除尺度不合理的框
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        # 同resize的时候相同的方式计算ratio
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        # 将预测的box按照宽高的变化量进行转换
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)将预测的bbox中超出原图的部分裁掉
        # 说明p_pred中的框是不是偏移量了，而是相对于图片来说
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)将无效bbox的coor置为0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes
```