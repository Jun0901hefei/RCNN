import numpy as np
import torch
import IOU
import predict
def nms(boxes, scores, iou_threshold):
    """
    输出的是置信度最高的框和与之不重叠的框
    :param boxes:所有预测边界框的坐标(左上右下)
    :param scores:每个预测框对应的置信度
    :param iou_threshold:如果两个框的 IoU 大于该阈值，则认为它们过于相似，需要抑制其中置信度较低的
    :return:保留下来的预测框在原始 boxes 中的索引列表
    """
    #按置信度从高到低排列的框的索引
    B = torch.argsort(scores, dim=-1, descending=True)
    #保留预测边界框的指标
    keep = []
    while B.size > 0:
        #取出当前置信度最高的框
        i = B[0]
        keep.append(i)
        if B.size == 1:
            break
        #从B[1：]与B[0]的IOU值
        iou = IOU.box_iou(
            boxes[i, :].reshape(-1, 4),
            boxes[B[1:], :].reshape(-1, 4)
        ).reshape(-1)
        #取出iou列表中小于阈值的索引
        index = np.nonzero(iou <= iou_threshold)[0]
        B = B[index + 1]
    return torch.tensor(keep, device=boxes.device)
def multi_box_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """
    :param cls_probs:(batch_size, num_classes, num_anchors)每个锚框属于每个类别的概率
    :param offset_preds:(batch_size, num_anchors * 4)每个锚框的4个偏移量预测
    :param anchors:(1, num_anchors, 4)所有锚框的坐标
    :param nms_threshold:NMS 的 IoU 阈值，默认 0.5
    :param pos_threshold:置信度阈值，低于此值视为背景，默认 0.009999999
    :return:输出每个锚框的[类别, 置信度, x1, y1, x2, y2]
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        #cls_prob当前batch的每个锚框属于每个类别的概率（num_classes, num_anchors)
        #offset_pred当前batch的每个锚框的4个偏移量预测（num_anchors，4）
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        #conf每一列的最大值
        #class_id每一个anchor对应的label
        conf, class_id = torch.max(cls_prob[1:], 0)
        #预测的边界框(num_anchors, 4)
        predicted_bb = predict.offset_inverse(anchors, offset_pred)
        #将预测的边界框经过非极大值抑制，得到保留下来的预测框的索引
        keep = nms(predicted_bb, conf, nms_threshold)
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        #uniques不重复值的列表
        #counts每个值对应的次数
        uniques, counts = combined.unique(return_counts=True)
        #未保留的预测狂的索引
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        #重新排序，前面全是保留下的锚框，后面全是背景
        class_id,conf,predicted_bb = class_id[all_id_sorted],conf[all_id_sorted],predicted_bb[all_id_sorted]
        #找出置信度小于阈值的索引
        below_min_idx = (conf < pos_threshold)
        #把这些全部变成背景
        class_id[below_min_idx] = -1
        #这些框当预测框置信度很小，那么当背景置信度就很大
        conf[below_min_idx] = 1 - conf[below_min_idx]
        #每个锚框的[类别, 置信度, x1, y1, x2, y2]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    #(batch_size, num_anchors, 6)
    return torch.stack(out)
