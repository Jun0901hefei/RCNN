import torch
import corner_and_center
import assign_anchor_box
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    计算每个锚框与其对应的标准框的偏移量

    假设一共有n个锚框，其中分配到标准框的有m个，那么anchors和assigned_bb的形状都为（m，4）

    :param anchors: 锚框的左上右下坐标
    :param assigned_bb:每个锚框匹配的真实边界框的左上右下坐标
    :param eps:一个很小的数，防止除以零或取对数时出现数值不稳定
    :return:每个锚框相对于其匹配的真实框的偏移量。
    """
    #将锚框从角点格式转换为中心格式
    c_anc = corner_and_center.box_corner_to_center(anchors)
    # 将标准框从角点格式转换为中心格式
    c_assigned_bb = corner_and_center.box_corner_to_center(assigned_bb)
    #计算中心坐标的偏移量
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    #计算高宽的偏移量
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    #合并成[x的偏移量，y的偏移量，w的偏移量，h的偏移量]
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset
def multi_box_target(anchors, labels):
    """
    :param anchors: 锚框的左上右下坐标
    (batch_size, num_anchors, 4)，通常batch_size为1，所有图片共用一套锚框
    :param labels: 标签(batch_size, num_gt_boxes, 5)，最后一维的 5 个值分别是：[类别标签, x_min, y_min, x_max, y_max]
    :return:
    bbox_offset：形状 (batch_size, num_anchors * 4)，所有锚框的偏移量预测目标（负样本的偏移量为 0）
    bbox_mask：形状 (batch_size, num_anchors * 4)，掩码矩阵，正样本对应的位置为 1，负样本为 0
    class_labels：形状 (batch_size, num_anchors)，每个锚框的类别标签（0 表示背景，>0 表示具体类别）
    """
    #取出batch_size，如果anchors的batch_size为1，那么就把第一维去掉
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    #记录每一个batch的偏移量，掩码，锚框分配的类别标签
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        #每个锚框分配的标准框在label里面的index 或者-1（背景）
        anchors_bbox_map = assign_anchor_box.assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        #将被分配到标准框的锚框设成1，背景设成0，并且转换成(num_anchors, 4)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        #初始化每个锚框分配到的label为0
        class_labels = torch.zeros(num_anchors, dtype=torch.long,device=device)
        #初始化每个锚框分配到的label的坐标为(num_anchors, 4)全0
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,device=device)
        #找到所有正样本锚框的索引
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        #找到所有正样本锚框对应的真实框在label里面的索引
        bb_idx = anchors_bbox_map[indices_true]
        #将分配到的真实框label+1，然后将其分配给对应的锚框，背景依然是0
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        #为正样本锚框设置对应的真实框坐标
        assigned_bb[indices_true] = label[bb_idx, 1:]
        #计算每个anchor和bbox的偏差，×bbox_mask是为了将没有分配到标准框的锚框偏差值变成0
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask

        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels_ = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels_)