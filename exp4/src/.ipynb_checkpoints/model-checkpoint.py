import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(model_name, num_classes=2, is_pretrained=True, keep_coco_head=False):
    """
    keep_coco_head=True: 返回原汁原味的 COCO 模型 (91类)，不替换头
    keep_coco_head=False: 替换为我们的 num_classes (2类)
    """
    print(f"🔄 加载模型: {model_name} (Pretrained={is_pretrained}, COCO_Head={keep_coco_head})...")
    
    # 1. 选择基础模型
    if model_name == 'mb3_320':
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if is_pretrained else None
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    elif model_name == 'mb3_fpn':
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if is_pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    elif model_name == 'resnet50':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if is_pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        raise ValueError(f"未知模型: {model_name}")

    # 2. 关键判断：如果要保留原装头，直接返回！
    if keep_coco_head:
        return model

    # 3. 否则，替换头 (用于微调)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model