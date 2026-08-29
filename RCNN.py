import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from matplotlib import pyplot as plt, patches
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import anchor_box_generator
import assign_anchor_box
import corner_and_center
import data_precession
import IOU
import Mark_categories_and_offsets
import NMSandMulti_box_detection
import predict
import show_boundingbox
class R_CNN:
    def __init__(self, num_classes, device='cuda'):
        """
        初始化R_CNN模型
        :param num_classes: 目标类别数（不包括背景）
        :param device: 设备
        """
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # 锚框参数
        self.sizes = [0.75]
        self.ratios = [1]
        self.boxes_per_pixel = len(self.sizes) + len(self.ratios) - 1
        # CNN特征提取器（AlexNet）
        self.cnn = None
        self.feature_extractor = None
        # SVM分类器列表（每个类别一个二分类SVM）
        self.svm_classifiers = []
        # 特征标准化器
        self.scaler = None
        # 边界框回归器（线性模型）
        self.bbox_regressor = None
    def load_pretrained_Alexnet(self):
        """
        加载预训练的AlexNet，并修改最后一层用于微调
        """
        # 加载预训练的AlexNet
        self.cnn = models.alexnet(pretrained=True)
        # 替换分类器的最后一层
        num_features = self.cnn.classifier[6].in_features
        self.cnn.classifier[6] = nn.Linear(num_features, self.num_classes + 1)
        self.cnn = self.cnn.to(self.device)
    def generate_anchors_for_image(self, image):
        """
        为单张图片生成锚框
        :param image: 图片张量 (C, H, W)
        :return: 锚框坐标 (num_anchors, 4)
        """
        # 确保图片在GPU上
        if not image.is_cuda:
            image = image.to(self.device)
        # 生成锚框
        anchors = anchor_box_generator.anchor_box_(
            image,  # 添加batch维度
            sizes=self.sizes,
            ratios=self.ratios
        )
        # # 每 10个像素取一个锚框（65536 → 16384）
        # stride = 256
        # if anchors.shape[0] > 10000:
        #     anchors = anchors[::stride]
        return anchors
    def extract_anchor_features(self, image, anchors):
        """
        从锚框区域提取4096维特征（RCNN风格：每个锚框单独通过CNN）
        :param image: 原始图片 (C, H, W)
        :param anchors: 锚框列表 (N, 4) [x1, y1, x2, y2]
        :return: 特征矩阵 (N, 4096)
        """
        self.feature_extractor.eval()
        #储存每个锚框的特征向量的列表
        features = []
        img_h, img_w = image.shape[1], image.shape[2]
        # 确保 anchors 是 2D
        if anchors.dim() == 1:
            anchors = anchors.unsqueeze(0)
        with (torch.no_grad()):
            for anchor in anchors:
                # 将锚框坐标转换为整数
                x1 = int(anchor[0].item() * img_w)
                y1 = int(anchor[1].item() * img_h)
                x2 = int(anchor[2].item() * img_w)
                y2 = int(anchor[3].item() * img_h)
                # 确保在有效范围内
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(1, min(x2, img_w))
                y2 = max(1, min(y2, img_h))
                if x1 >= x2 or y1 >= y2:
                    continue
                # 裁剪锚框区域
                anchor_crop = image[:, y1:y2, x1:x2]
                # 调整大小为224x224（AlexNet期望的输入尺寸）
                anchor_resized = torch.nn.functional.interpolate(
                    anchor_crop.unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False)
                # 提取特征
                feat = self.feature_extractor(anchor_resized)
                features.append(feat.squeeze().cpu().numpy())
        return np.array(features)#为什么用np格式，是因为svm只支持np格式
    def _build_feature_extractor(self):
        """
        构建特征提取器（从当前的 self.cnn 创建）
        去掉最后一层，输出4096维特征
        """
        self.feature_extractor = nn.Sequential(
            *list(self.cnn.features.children()),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            *list(self.cnn.classifier.children())[:-1]  # 去掉最后一层
        )
        self.feature_extractor = self.feature_extractor.to(self.device)
    def fine_tune_cnn(self, train_loader, num_epochs=1,patience=1,lr=0.001, min_delta=0.01):
        """
        使用锚框（IoU>0.5为正样本）微调CNN
        :param train_loader: 训练数据加载器
        :param num_epochs: 训练轮数
        :param min_delta: loss下降的最小阈值，小于此值视为没有改进
        :param patience: 连续多少个epoch loss不下降则停止
        :param lr: 学习率
        """
        print("Starting CNN fine-tuning...")
        # 所有层都可训练（或选择性解冻）
        for param in self.cnn.parameters():
            param.requires_grad = True
        # 设置优化器和损失函数
        optimizer = optim.SGD(self.cnn.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        # 早停相关变量
        best_loss = float('inf')
        patience_counter = 0
        early_stop = False
        # 训练循环
        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            self.cnn.train()
            total_loss = 0
            total_samples = 0
            for batch_idx, (images, labels) in enumerate(
                    tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.shape[0]
                all_images = []
                all_class_labels = []
                for i in range(batch_size):
                    # 生成锚框
                    anchors = self.generate_anchors_for_image(images[i])
                    # 获取当前图像的标签
                    img_labels = labels[i]
                    #防御性编程，如果这个label没有数据，就跳出
                    if len(img_labels) == 0:
                        continue
                    #如果只有一维增加一维
                    if img_labels.dim() == 1:
                        img_labels=img_labels.unsqueeze(0)
                    # 分配锚框到真实框
                    anchors_bbox_map = assign_anchor_box.assign_anchor_to_bbox(
                        img_labels[:, 1:], anchors, self.device, iou_threshold=0.5
                    )
                    # 选择正样本（IoU > 0.5）
                    positive_mask = anchors_bbox_map >= 0
                    positive_indices = torch.nonzero(positive_mask).squeeze()
                    # 确保 positive_indices 是 1D
                    if positive_indices.dim() == 0:
                        positive_indices = positive_indices.unsqueeze(0)
                    #如果这张图片没有正样本，那就跳出
                    if positive_indices.numel() == 0:
                        continue
                    # 获取正样本锚框和对应的类别标签
                    positive_anchors = anchors[positive_indices]
                    positive_labels = anchors_bbox_map[positive_indices]
                    # 确保 positive_anchors 是 2D
                    if positive_anchors.dim() == 1:
                        positive_anchors = positive_anchors.unsqueeze(0)
                    # 获取每个锚框对应的真实类别
                    class_labels = img_labels[positive_labels, 0].long() + 1
                    # 确保 class_labels 是 1D
                    if class_labels.dim() == 0:
                        class_labels = class_labels.unsqueeze(0)
                    img_h, img_w = images[i].shape[1], images[i].shape[2]
                    for idx,anchor in enumerate(positive_anchors):
                        x1 = int(anchor[0].item() * img_w)
                        y1 = int(anchor[1].item() * img_h)
                        x2 = int(anchor[2].item() * img_w)
                        y2 = int(anchor[3].item() * img_h)
                        # 确保在有效范围内
                        x1 = max(0, min(x1, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        x2 = max(1, min(x2, img_w))
                        y2 = max(1, min(y2, img_h))
                        if x1 >= x2 or y1 >= y2:
                            continue
                        anchor_crop = images[i][:, y1:y2, x1:x2]
                        anchor_resized = torch.nn.functional.interpolate(
                            anchor_crop.unsqueeze(0),
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        )
                        all_images.append(anchor_resized)
                        all_class_labels.append(class_labels[idx])
                    if len(all_images) == 0:
                        continue
                        # 合并所有锚框
                    images_batch = torch.cat(all_images, dim=0)  # (N, 3, 224, 224)
                    labels_batch = torch.tensor(all_class_labels, dtype=torch.long).to(self.device)
                    # 前向传播
                    outputs = self.cnn(images_batch)
                    loss = criterion(outputs, labels_batch)
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_samples+=images_batch.size(0)
                    total_loss += loss.item()
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')
            # 早停检查
            if avg_loss < best_loss - min_delta:
                # Loss 有显著下降
                best_loss = avg_loss
                patience_counter = 0
                print(f"  ✓ Loss improved to {best_loss:.4f}")
            else:
                # Loss 没有显著下降
                patience_counter += 1
                print(f"  ✗ No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    early_stop = True
                    print(f"  Early stopping triggered!")
        print("CNN fine-tuning completed!")
        # 冻结CNN参数
        for param in self.cnn.parameters():
            param.requires_grad = False
        # 用微调后的权重更新特征提取器
        self._build_feature_extractor()
        self.feature_extractor.eval()
    def prepare_training_data(self, train_loader, iou_threshold=0.6, bg_iou_threshold=0.4):
        """
        准备SVM和边界框回归器的训练数据
        :param train_loader: 训练数据加载器
        :param iou_threshold: IoU阈值
        :return: 特征矩阵，类别标签，偏移量标签
        """
        print("Preparing training data for SVM and bbox regressor...")
        all_features = []
        all_class_labels = []
        all_offset_labels = []
        for images, labels in tqdm(train_loader, desc='Preparing training data'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.shape[0]
            for i in range(batch_size):
                anchors = self.generate_anchors_for_image(images[i])
                img_labels = labels[i]
                if len(img_labels) == 0:
                    continue
                if img_labels.dim() == 1:
                    img_labels = img_labels.unsqueeze(0)
                # 分配锚框到真实框
                anchors_bbox_map = assign_anchor_box.assign_anchor_to_bbox(
                    img_labels[:, 1:], anchors, self.device, iou_threshold=iou_threshold
                )
                num_anchors = anchors.shape[0]
                # 为所有锚框分配标签（包括背景）
                all_anchors_labels = torch.zeros(num_anchors, dtype=torch.long, device=self.device)
                all_anchors_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=self.device)
                # 正样本：IoU > 0.5
                positive_mask = anchors_bbox_map >= 0
                positive_indices = torch.nonzero(positive_mask).squeeze()
                if positive_indices.numel() > 0:
                    # 确保是 1D（如果只有一个元素，dim() == 0）
                    positive_indices = positive_indices.reshape(-1)  # 自动变为 1D

                    positive_labels = anchors_bbox_map[positive_indices]
                    all_anchors_labels[positive_indices] = img_labels[positive_labels, 0].long() + 1
                    all_anchors_bb[positive_indices] = img_labels[positive_labels, 1:5]
                # 负样本（背景）：IoU < bg_iou_threshold 的锚框
                # 计算所有锚框与所有真实框的IOU
                jaccard = IOU.box_iou(anchors, img_labels[:, 1:])
                max_iou, _ = torch.max(jaccard, dim=1)
                # 背景：最大IOU < bg_iou_threshold
                bg_mask = max_iou < bg_iou_threshold
                bg_indices = torch.nonzero(bg_mask).squeeze()
                if bg_indices.numel() > 0:
                    # 确保是 1D（如果只有一个元素，dim() == 0）
                    bg_indices = bg_indices.reshape(-1)  # 自动变为 1D
                    all_anchors_labels[bg_indices] = 0
                # 忽略样本：bg_iou_threshold <= IoU <= iou_threshold 的样本不参与训练
                # 这些样本不需要标记，因为已经初始化为0，但我们需要排除它们
                # 方法：将忽略样本的标签设为 -1（在后续处理中跳过）
                ignore_mask = (max_iou >= bg_iou_threshold) & (max_iou < iou_threshold)
                ignore_indices = torch.nonzero(ignore_mask).squeeze()
                if ignore_indices.numel() > 0:
                    ignore_indices = ignore_indices.reshape(-1)
                    all_anchors_labels[ignore_indices] = -1  # -1 表示忽略
                # 收集所有需要训练的样本（正样本 + 背景样本）
                train_mask = all_anchors_labels != -1
                train_indices = torch.nonzero(train_mask).squeeze().reshape(-1)
                if train_indices.numel() == 0:
                    continue
                # 采样背景样本（避免类别不平衡）
                bg_train_indices = torch.nonzero(all_anchors_labels == 0).squeeze().reshape(-1)
                positive_train_indices = torch.nonzero(all_anchors_labels > 0).squeeze().reshape(-1)
                # 限制背景样本数量（不超过正样本数量的3倍）
                if bg_train_indices.numel() > 0 and positive_train_indices.numel() > 0:
                    max_bg_samples = positive_train_indices.numel() * 3
                    if bg_train_indices.numel() > max_bg_samples:
                        perm = torch.randperm(bg_train_indices.numel(), device=self.device)
                        bg_train_indices = bg_train_indices[perm[:max_bg_samples]]
                        # 合并正样本和采样后的背景样本
                final_indices = torch.cat([positive_train_indices,
                                                   bg_train_indices]) if positive_train_indices.numel() > 0 else bg_train_indices
                if final_indices.numel() == 0:
                    continue
                # 提取特征
                selected_anchors = anchors[final_indices]
                selected_labels = all_anchors_labels[final_indices]
                anchor_features = self.extract_anchor_features(images[i], selected_anchors)
                all_features.append(anchor_features)
                all_class_labels.append(selected_labels.cpu().numpy())
                # 计算偏移量（仅正样本需要）
                offsets = torch.zeros((final_indices.numel(), 4), dtype=torch.float32, device=self.device)
                for j, idx in enumerate(final_indices):
                    if all_anchors_labels[idx] > 0:  # 正样本
                        # 获取对应的真实框
                        assigned_bb = all_anchors_bb[idx].unsqueeze(0)
                        anchor = anchors[idx].unsqueeze(0)
                        offsets[j] = Mark_categories_and_offsets.offset_boxes(anchor, assigned_bb)
                all_offset_labels.append(offsets.cpu().numpy())
            # 合并所有数据
        X = np.concatenate(all_features, axis=0)
        y_class = np.concatenate(all_class_labels, axis=0)
        y_offset = np.concatenate(all_offset_labels, axis=0)

        print(f"Prepared {X.shape[0]} training samples")
        unique, counts = np.unique(y_class, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

        return X, y_class, y_offset
    def train_svm(self, X, y, kernel='linear', C=1.0):
        """
        训练SVM分类器（每个类别一个二分类SVM，包括背景类）
        :param X: 特征矩阵 (N, 4096)
        :param y: 类别标签 (N,)
        :param kernel: SVM核函数
        :param C: 正则化参数
        """
        print("Training SVM classifiers...")
        # 特征标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        # 初始化分类器列表，长度为 num_classes
        self.svm_classifiers = [None] * (self.num_classes+1)
        for class_id in range(0, self.num_classes + 1):
            # 构建二分类标签：当前类为正，其他为负
            y_binary = np.where(y == class_id, 1, 0)
            # 检查是否有正样本
            if np.sum(y_binary) == 0:
                print(f"Warning: No positive samples for class {class_id}")
                # 创建占位符分类器（预测时始终返回 0）
                self.svm_classifiers[class_id] = None
                continue
            # 训练SVM
            clf = svm.SVC(kernel=kernel, C=C, probability=True, random_state=42)
            clf.fit(X_scaled, y_binary)
            self.svm_classifiers[class_id] = clf
            print(f'SVM for class {class_id} trained')
        print(f"SVM training completed! Trained {len(self.svm_classifiers)} classifiers")
    def train_bbox_regressor(self, X, y_offset, lr=0.001, num_epochs=100):
        """
        训练边界框回归器（线性回归）
        :param X: 特征矩阵 (N, 4096)
        :param y_offset: 偏移量标签 (N, 4)
        :param lr: 学习率
        :param num_epochs: 训练轮数
        """
        print("Training bbox regressor...")
        # 特征标准化
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        # 转换为张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_offset, dtype=torch.float32).to(self.device)
        # 线性模型
        self.bbox_regressor = nn.Linear(X_scaled.shape[1], 4)
        self.bbox_regressor.to(self.device)
        # 优化器和损失函数
        optimizer = optim.Adam(self.bbox_regressor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        # 训练
        self.bbox_regressor.train()
        for epoch in range(num_epochs):
            # 打乱数据
            indices = torch.randperm(X_tensor.shape[0])
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            # 前向传播
            outputs = self.bbox_regressor(X_shuffled)
            loss = criterion(outputs, y_shuffled)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')
        self.bbox_regressor.eval()
        print("Bbox regressor training completed!")
    def fit(self, train_loader, num_epochs=10, lr=0.001):
        """
        完整训练流程
        :param train_loader: 训练数据加载器
        :param num_epochs: CNN微调轮数
        :param lr: 学习率
        """
        # 1. 加载预训练的AlexNet
        self.load_pretrained_Alexnet()
        # 2. 微调CNN
        self.fine_tune_cnn(train_loader, num_epochs=num_epochs, lr=lr)
        # 3. 准备训练数据
        X, y_class, y_offset = self.prepare_training_data(train_loader)
        # 4. 训练SVM
        self.train_svm(X, y_class)
        # 5. 训练边界框回归器
        self.train_bbox_regressor(X, y_offset)
        print("RCNN training completed!")
    def predict(self, image, anchors=None, nms_threshold=0.5, conf_threshold=0.5):
        """
        目标检测预测
        :param image: 输入图片 (C, H, W)
        :param anchors: 预生成的锚框（可选）
        :param nms_threshold: NMS阈值
        :param conf_threshold: 置信度阈值
        :return: 检测结果 [类别, 置信度, x1, y1, x2, y2]
        """
        # 确保图片在GPU上
        if not image.is_cuda:
            image = image.to(self.device)
        # 生成锚框
        if anchors is None:
            anchors = self.generate_anchors_for_image(image)
        # 提取所有锚框的特征
        anchor_features = self.extract_anchor_features(image, anchors)
        # 使用SVM预测类别
        anchor_features_scaled = self.scaler.transform(anchor_features)
        class_scores = []
        for class_id in range(0, self.num_classes + 1):
            clf = self.svm_classifiers[class_id]
            if clf is None:
                # 占位符：所有样本置信度为0
                scores = np.zeros(anchor_features_scaled.shape[0])
            else:
                # 取出所有正样本的置信度
                scores = clf.predict_proba(anchor_features_scaled)[:, 1]
            class_scores.append(scores)
        # 转换为 PyTorch 张量，格式为 (batch_size, num_classes, num_anchors)
        cls_probs = torch.tensor(class_scores, dtype=torch.float32).unsqueeze(0).to(self.device)
        # cls_probs shape: (1, num_classes, num_anchors)
        # 使用边界框回归器修正锚框
        with torch.no_grad():
            offset_pred = self.bbox_regressor(
                torch.tensor(anchor_features_scaled, dtype=torch.float32).to(self.device)
            )
        # 准备 multi_box_detection 需要的格式
        anchors_tensor = anchors.unsqueeze(0).to(self.device)  # (1, num_anchors, 4)
        offset_preds = offset_pred.reshape(1, -1)  # (1, num_anchors * 4)
        results = NMSandMulti_box_detection.multi_box_detection(
            cls_probs,  # (1, num_classes, num_anchors)
            offset_preds,  # (1, num_anchors * 4)
            anchors_tensor,  # (1, num_anchors, 4)
            nms_threshold=nms_threshold,
            pos_threshold=conf_threshold
        )
        # results shape: (1, num_anchors, 6)
        # 每行: [类别, 置信度, x1, y1, x2, y2]
        # 转换为 numpy 并去除 batch 维度
        results = results.squeeze(0).cpu().numpy()
        # 过滤掉背景（类别为 -1）
        results = results[results[:, 0] != -1]
        return results

    def save_model(self, save_dir='./rcnn_models'):
        """
        保存训练好的模型
        """
        os.makedirs(save_dir, exist_ok=True)
        # 保存CNN参数
        torch.save(self.cnn.state_dict(), os.path.join(save_dir, 'cnn.pth'))
        # 保存SVM分类器
        with open(os.path.join(save_dir, 'svm_classifiers.pkl'), 'wb') as f:
            pickle.dump(self.svm_classifiers, f)
        # 保存标准化器
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            # 保存边界框回归器
        torch.save(self.bbox_regressor.state_dict(), os.path.join(save_dir, 'bbox_regressor.pth'))
        # 保存配置
        config = {
            'num_classes': self.num_classes,
            'sizes': self.sizes,
            'ratios': self.ratios,
            'boxes_per_pixel': self.boxes_per_pixel
        }
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)

        print(f"Model saved to {save_dir}")
    def load_model(self, save_dir='./rcnn_models'):
        """
        加载训练好的模型
        """
        # 加载配置
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        self.num_classes = config['num_classes']
        self.sizes = config['sizes']
        self.ratios = config['ratios']
        self.boxes_per_pixel = config['boxes_per_pixel']
        # 加载CNN
        self.load_pretrained_Alexnet()
        self.cnn.load_state_dict(torch.load(os.path.join(save_dir, 'cnn.pth')))
        # 加载SVM
        with open(os.path.join(save_dir, 'svm_classifiers.pkl'), 'rb') as f:
            self.svm_classifiers = pickle.load(f)
        # 加载标准化器
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        # 加载边界框回归器
        self.bbox_regressor = nn.Linear(4096, 4)
        self.bbox_regressor.load_state_dict(torch.load(os.path.join(save_dir, 'bbox_regressor.pth')))
        self.bbox_regressor.to(self.device)
        self.bbox_regressor.eval()
        # 重建特征提取器
        self._build_feature_extractor()
        self.feature_extractor.eval()
        print(f"Model loaded from {save_dir}")
    def visualize_predictions(self,images, predictions, nrows=2, ncols=4, conf_threshold=0.5):
        """
        可视化预测结果
        :param images: 图片列表 (N, C, H, W)
        :param predictions: 预测结果列表/数组 (N, N_pred, 6)
                        每个预测: [类别, 置信度, xmin, ymin, xmax, ymax]
        :param nrows: 行数
        :param ncols: 列数
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
        #展平方便遍历
        axes = axes.flatten() if nrows * ncols > 1 else [axes]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

        for i, ax in enumerate(axes):
            if i >= len(images):
                ax.axis('off')
                continue
            # 显示图片
            img = images[i].permute(1, 2, 0).cpu().numpy()
            #将像素归一化
            img = img / img.max() if img.max() > 1.0 else img
            ax.imshow(img)
            # 获取预测
            preds = predictions[i] if i < len(predictions) else []
            # 找每个类别的最大置信度框
            best_per_class = {}
            for pred in preds:
                cid, conf, x1, y1, x2, y2 = pred[:6]
                if conf >= conf_threshold:
                    #如果这个类别在best_per_class还没出现，或者当前类别有置信度更高的锚框出现的时候
                    if cid not in best_per_class or conf > best_per_class[cid][0]:
                        best_per_class[cid] = (conf, [x1, y1, x2, y2])
            for idx, (cid, (conf, bbox)) in enumerate(best_per_class.items()):
                x1, y1, x2, y2 = bbox
                color = colors[idx % len(colors)]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'C{cid}: {conf:.2f}',
                        color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.7))
            ax.set_title(f'Image {i + 1}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    def evaluate_rcnn(self,test_loader, iou_threshold=0.5):
        """
        评估RCNN性能
        :param test_loader: 测试数据加载器
        :param iou_threshold: IoU阈值
        :return: precision, recall
        """
        print("\nEvaluating RCNN...")
        total_detections = 0
        total_true_positives = 0
        total_gt_boxes = 0
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            for i in range(images.shape[0]):
                img = images[i]
                gt_boxes = labels[i][:, 1:5].cpu().numpy()  # [xmin, ymin, xmax, ymax]
                gt_classes = labels[i][:, 0].long().cpu().numpy() + 1  # 类别+1
                # 累加真实框总数
                total_gt_boxes += len(gt_boxes)
                preds = self.predict(img, nms_threshold=0.5, conf_threshold=0.5)
                # 累加预测框总数
                total_detections += len(preds)
                # 如果没有预测框，跳过匹配
                if len(preds) == 0 or len(gt_boxes) == 0:
                    continue
                # 将预测框转换为张量
                pred_boxes = torch.tensor([p[2:6] for p in preds], dtype=torch.float32).to(self.device)
                pred_classes = torch.tensor([p[0] for p in preds], dtype=torch.long).to(self.device)
                # 计算所有 IOU（向量化）
                gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
                pred_to_gt_map = assign_anchor_box.assign_anchor_to_bbox(
                    gt_boxes_tensor,    # ground_truth
                    pred_boxes,         # anchors（这里传入预测框）
                    self.device,
                    iou_threshold
                )
                for j in range(len(preds)):
                    gt_idx = pred_to_gt_map[j].item()
                    if gt_idx != -1 and pred_classes[j] == gt_classes[gt_idx]:
                        total_true_positives += 1
        precision = total_true_positives / total_detections if total_detections > 0 else 0
        recall = total_true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

def main():
    """
    主函数：训练和测试R-CNN
    """
    # 参数设置
    batch_size = 1
    num_classes = 1  # 香蕉数据集只有1个类别
    model_dir = './rcnn_models'
    # 加载数据
    print("Loading data...")
    train_loader, test_loader = data_precession.load_data_bananas(batch_size)
    # 初始化R-CNN
    rcnn = R_CNN(num_classes=num_classes, device='cuda')
    # 训练
    # 检查是否有训练好的模型
    if os.path.exists(os.path.join(model_dir, 'config.pkl')):
        print(f"Found existing model at {model_dir}, loading...")
        rcnn.load_model(model_dir)
        print("Model loaded successfully!")
    else:
        print("No existing model found. Training from scratch...")
        rcnn.fit(train_loader, num_epochs=10, lr=0.001)
        rcnn.save_model(model_dir)
    # 测试
    print("\nTesting R-CNN...")
    test_images = []
    test_predictions = []
    max_images = 2
    for images, labels in test_loader:
        for j in range(images.shape[0]):
            if len(test_images) >= max_images:
                break
            img = images[j].to(rcnn.device)
            preds = rcnn.predict(img, nms_threshold=0.5, conf_threshold=0.5)
            test_images.append(img)
            test_predictions.append(preds)
        if len(test_images) >= max_images:
            break
    print("\nVisualizing predictions...")
    rcnn.visualize_predictions(test_images, test_predictions, nrows=2, ncols=4)
    # 评估
    rcnn.evaluate_rcnn(test_loader)
if __name__ == '__main__':
    main()