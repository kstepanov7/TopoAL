# -*- coding: utf-8 -*-

import torch
from TopoAL.src.utils import STE, Dilation

def run_epoch(model, optimizer, criterion, dataloader, device, metric=None):

    epoch_loss, epoch_metric = 0.0, 0.0
    with torch.set_grad_enabled(optimizer is not None):        
        for image, mask_gt in dataloader:

          image, mask_gt = image.to(device), mask_gt.to(device)

          pred = model(image).sigmoid()
          loss = criterion(pred, mask_gt.float())
          epoch_loss += loss.detach().cpu().numpy()

          if optimizer is not None:
              optimizer.zero_grad()
              loss.backward() 
              optimizer.step()
          
          if metric != None:
              epoch_metric += metric(pred.detach().cpu(), mask_gt.detach().cpu())
    
    if metric != None:
        return epoch_loss/len(dataloader), epoch_metric/len(dataloader)
    else:
        return epoch_loss/len(dataloader)



def run_epoch_topo(generator, discriminator, optimizer_G, optimizer_D, criterion_G, criterion_D, dataloader, device, metric=None):
    epoch_loss_G, epoch_loss_D = 0, 0
    epoch_metric = 0
    with torch.set_grad_enabled(optimizer_G is not None):        
        for image, mask_gt in dataloader:
          image, mask_gt = image.to(device), mask_gt.to(device)

          '''Train Discriminator'''
          pred = generator(image).sigmoid()

          # generate input for D
          pred_tr = STE()(pred)
          gt_dil = Dilation(mask_gt, dil=2, device=device)
          T_0 = gt_dil * pred_tr
          T = torch.cat([T_0, image], 1)

          # generate labels for D
          labels_D = discriminator.assign_labels(mask_gt.detach().cpu().float(), pred.detach().cpu().float(), sigma=5)

          # make pred by D
          pred_D = discriminator(T)
          pred_xy = discriminator(torch.cat([mask_gt, image], 1))
                  
          # calc loss for D
          loss_D = criterion_D(pred_D, labels_D, pred_xy)

          if optimizer_D is not None:
              optimizer_D.zero_grad()
              loss_D.backward(retain_graph=True)
              optimizer_D.step()


          '''Train Generator'''

          pred_g = generator(image).sigmoid()
          pred_tr = STE()(pred_g)
          T_0 = Dilation(mask_gt, device=device) * pred_tr
          T = torch.cat([T_0, image], 1)
          pred_D = discriminator(T)

          # calc loss for G
          loss_G = criterion_G(mask_gt.float(), pred_g, pred_D)

          if optimizer_G is not None:
              optimizer_G.zero_grad()
              loss_G.backward()
              optimizer_G.step()

          # log stats
          epoch_loss_G += loss_G.detach().cpu().numpy()
          epoch_loss_D += loss_D.detach().cpu().numpy()
          epoch_metric += metric(pred_g.detach().cpu().numpy(), mask_gt.detach().cpu().numpy())

    return epoch_loss_G/len(dataloader), epoch_loss_D/len(dataloader), epoch_metric/len(dataloader)
