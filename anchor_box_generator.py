import torch
import matplotlib.patches as patches
from matplotlib import pyplot as plt
def anchor_box_(data, sizes, ratios):
    """
    生成以每个像素为中心具有不同形状的锚框
    :param data: 导入的单张图片（c*h*w）
    :param sizes: 生成的锚框的占原图的比例的列表
    :param ratios: 生成的锚框的高宽比的列表
    :return: 所有锚框的左上右下坐标(num_anchors, 4)
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    #每个像素的锚框数
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    #每个像素的中心点偏移量
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长
    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    #锚框的宽度
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    #锚框的高度
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    #生成[[-w1/2,-h1/2,w1/2,h1/2],
    #    [-w2/2,-h2/2,w2/2,h2/2],
    #    ...,
    #    [-w1/2,-h1/2,w1/2,h1/2],
    #    [-w2/2,-h2/2,w2/2,h2/2]]
    # 一共像素个
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2
    #生成[[x1,y1,x1,y1],
    #   [x1,y1,x1,y1],
    #   [x2,y2,x2,y2],
    #   [x2,y2,x2,y2],
    #   ...]
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    #生成每个anchor_box的左上右下坐标
    output = out_grid + anchor_manipulations
    return output
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
    展示固定像素点的所有锚框
    :param axes:画板
    :param bboxes:要显示的边界框列表
    :param labels:与边界框对应的标签列表
    :param colors:边界框的颜色列表
    """
    def _make_list(obj, default_values=None):
        """
        如果不是列表的话，变成列表
        :param obj:
        :param default_values:
        :return:
        """
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center',
                      ha='center',
                      fontsize=9,
                      color=text_color,
                      bbox=dict(facecolor=color, lw=0))

if __name__ == '__main__':

    img = plt.imread('people.jpg')
    img_gpu = torch.from_numpy(img).permute(2, 0, 1).cuda()
    Y = anchor_box_(img_gpu, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = Y.reshape(1280, 1920, 5, 4)
    fig = plt.imshow(img)
    bbox_scale = torch.tensor((1920, 1280,1920, 1280),device='cuda')
    selected_boxes = boxes[800, 600, :, :] * bbox_scale
    selected_boxes_cpu = selected_boxes.cpu()
    show_bboxes(fig.axes, selected_boxes_cpu,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                 's=0.75, r=0.5'])
    plt.show()
