import numpy as np
import corner_and_center
def offset_inverse(anchors, offset_pre):
    """
    将预测的锚框偏移量转化为预测锚框的左上右下坐标
    :param anchors: 锚框的左上右下坐标
    :param offset_pre: 预测的偏移量
    :return:预测锚框的左上右下坐标
    """
    anc = corner_and_center.box_corner_to_center(anchors)
    #真实中心点 = 锚框中心点 + （预测偏移量 × 锚框宽高 / 10）除以10是一个缩放因子，用来控制偏移量的范围，使训练更稳定
    pre_bbox_xy = (offset_pre[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    #真实宽高 = 锚框宽高 × exp(预测宽高偏移 / 5)
    pre_bbox_wh = np.exp(offset_pre[:, 2:] / 5) * anc[:, 2:]
    #拼接起来，形成完整的中心格式预测框
    pre_bbox = np.concatenate((pre_bbox_xy, pre_bbox_wh), axis=1)
    #将预测框从“中心格式”再转换回“角点格式”
    predicted_bbox = corner_and_center.box_center_to_corner(pre_bbox)
    return predicted_bbox