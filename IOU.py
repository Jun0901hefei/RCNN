import torch

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    #计算框的面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    #两个锚框左上坐标的最大值
    inter_upper_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # 两个锚框右下坐标的最小值
    inter_lower_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lower_rights - inter_upper_lefts).clamp(min=0)#当两个框完全不重合的时候，会出现负数，就赋值为0
    # inter_areas and union_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas