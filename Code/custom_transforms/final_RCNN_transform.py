from torchvision import transforms
import torch,cv2,numpy as np
import torchvision.transforms.functional as fn

# imageID, p_bboxes, gd_bboxes, labels, offsets = self.df.iloc[idx]

class FinalRCNNTransform:
    def __init__(self,all_labels):
        """all_labels : list of all labels ,should also include the background"""
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.pil_to_tensor = transforms.ToTensor()
        self.all_labels = all_labels
        self.label_to_target,self.target_to_label = self.get_label_to_target()

    def get_label_to_target(self):
        """
            the target is a numerical value to represent the label (usefull for cross-entropy
             loss)
             returns also the transformation from target to labels
        """

        label_to_target = {l: t for t, l in enumerate(self.all_labels)}
        target_to_label = {t: l for l, t in label_to_target.items()}
        assert label_to_target['Background'] == 0
        return label_to_target,target_to_label

    def preprocess_image(self,img):
        # img = self.tra
        # torchvision.transforms.Resize(244,244)
        img = fn.resize(img, (224, 224))
        img = self.pil_to_tensor(img)
        img = self.normalize(img)
        img = img.unsqueeze(0)
        return img
        # return img.to(device).float()


    def __call__(self,el):
        pil_img, p_bboxes, gd_bboxes, labels, offsets = el

        crops = [bbox.crop_image(pil_img) for bbox in p_bboxes]
        crops = [self.preprocess_image(crop) for crop in crops]

        labels_as_targets = [self.label_to_target[label] for label in labels]

        return crops,labels_as_targets,offsets


