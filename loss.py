import torch

def active_contour_loss(y_pred, image, weight=10):
  '''
  y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
  weight: scalar, length term weight.
  '''
  # arc-length item
  delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal (B, C, H-1, W) 
  delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical   (B, C, H,   W-1)
  
  delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
  delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
  delta_pred = torch.abs(delta_r + delta_c) 

  epsilon = 1e-8 # a parameter to avoid calculating the square root of zero
  lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) 
  
  # image subregion coherence
  C_1  = torch.ones_like(y_pred)
  C_0 = torch.zeros_like(y_pred)
  judge = torch.where(y_pred>0.5, C_1, C_0)
  image = torch.mean(image,dim = 1).unsqueeze(1)
  C_in = torch.sum(judge*image)/torch.sum(judge)
  C_out = torch.sum((1-judge)*image)/torch.sum(1-judge)


  region_in  = torch.mean( judge     * (image - C_in )**2 )
  region_out = torch.mean( (1-judge) * (image - C_out)**2 ) 
  region = region_in + region_out
  
  loss =  weight*lenth + region

  return loss