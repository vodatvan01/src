import torch
import torch.nn as nn
import torch.nn.functional as F

class MODNetLoss(nn.Module):
    def __init__(self, semantic_scale = 1.0,detail_scale=10.0, matte_scale=1.0):
        super(MODNetLoss, self).__init__()
        self.semantic_scale = semantic_scale
        self.detail_scale = detail_scale
        self.matte_scale = matte_scale
    def forward(self, pred_semantic, pred_detail, pred_matte, image, gt_matte, trimap):
        global blurer
        gt_semantic = F.interpolate(gt_matte, scale_factor=1/16, mode='bilinear')
        gt_semantic = blurer(gt_semantic.float())
        semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
        semantic_loss = self.semantic_scale * semantic_loss
        
        boundaries = (trimap < 0.5) + (trimap > 0.5)

        pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
        gt_detail = torch.where(boundaries, trimap, gt_matte)
        
        
        detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
        
        
        detail_loss = self.detail_scale * detail_loss
        pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
        matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
        matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
            + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
        matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
        matte_loss = self.matte_scale * matte_loss
        return semantic_loss, detail_loss, matte_loss
