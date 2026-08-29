import torch
import IOU
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框
    :param ground_truth: 真实边界框
    :param anchors: 生成的锚框
    :param device: 设备
    :param iou_threshold: iou的阈值
    :return:每个锚框分配的标准框在label里面的index 或者-1（背景）
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = IOU.box_iou(anchors, ground_truth)
    # 初始化为-1，准备给每个锚框都分配一个标准框
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 找到每个锚框的最大 IoU 和对应的真实框索引
    max_ious, indices = torch.max(jaccard, dim=1)
    # 找出 IoU >= 阈值的锚框
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 获取这些锚框对应的真实框索引
    box_j = indices[max_ious >= iou_threshold]
    # 分配
    anchors_bbox_map[anc_i] = box_j
    #全为-1的向量，之后强制将分配后的行列的iou赋值为-1
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        #全局最大的iou（拉平后的索引）
        max_idx = torch.argmax(jaccard)
        #第几个标准框
        box_idx = (max_idx % num_gt_boxes).long()
        #第几个锚框
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map