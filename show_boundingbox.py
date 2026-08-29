import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bbox_on_image(images, bboxes, color='red', linewidth=2,nrows=1,ncols=1):
    """
    在图像上画边界框
    image: 图片 (H, W, C) 格式，值域 [0, 1]
    bbox: [x1, y1, x2, y2] 左上角和右下角坐标
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            bbox = bboxes[i]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'Image {i + 1}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()