#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings

warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
import re
from collections import defaultdict
from sklearn.metrics import f1_score, average_precision_score
from configs.template import config
from datasets.nvGesture import NvGestureDataset
from models.RODModel import create_model
from utils.loss import mixup_criterion

from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
    mixup_data,
    compute_mAP
)

def compute_P_matrix(model, train_loader, config,state):
    model.eval()
    # ----- Calculate the feature variance -----
    if state =='train_depth':
        modules = [m for n, m in model.depth_encoder.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules = modules + [m for n, m in model.cls_d.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
    elif state =='train_rgb':
        modules = [m for n, m in model.rgb_encoder.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules = modules + [m for n, m in model.cls_r.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
    else :
        modules = [m for n, m in model.of_encoder.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules = modules + [m for n, m in model.cls_o.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
    handles = []
    for m in modules:
        handles.append(m.register_forward_hook(hook=model.compute_cov))
    for step, (rgb, of, depth, y) in enumerate(train_loader):
        rgb = rgb.float().cuda()
        of = of.cuda()
        depth = depth.cuda()
        y = y.cuda()
        input_list = []
        input_list.append(rgb), input_list.append(of), input_list.append(depth),input_list.append(state)
        model.forward(input_list)
    # ----- Calculate the modification matrixs -----
    model.get_P_matrix(modules, model.fea_in, config)
    for h in handles:
        h.remove()


def train_withP(epoch, train_loader, model, optimizer, logger,state):
    model.train()

    tl = Averager()
    ta = Averager()

    for step,(rgb,of,depth,y) in enumerate(train_loader):
        rgb = rgb.float().cuda()
        of = of.cuda()
        depth = depth.cuda()
        y = y.cuda()
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        if state =='train_rgb':
            rgb, targets_a, targets_b, lam = mixup_data(rgb, y, config['train']['mixup_alpha'])
        elif state == 'train_of':
            of, targets_a, targets_b, lam = mixup_data(of, y, config['train']['mixup_alpha'])
        else :
            depth, targets_a, targets_b, lam = mixup_data(depth, y, config['train']['mixup_alpha'])

        input_list = []
        input_list.append(rgb), input_list.append(of), input_list.append(depth),input_list.append(state)
        fea, o = model(input_list)

        loss_ori = mixup_criterion(criterion, o, targets_a, targets_b, lam)
        # ----- SAM optimization -----

        if state =='train_depth':
            disturb_params = list(model.depth_encoder.parameters())
            disturb_params_cls = list(model.cls_d.parameters())
        elif state == 'train_rgb':
            disturb_params = list(model.rgb_encoder.parameters())
            disturb_params_cls = list(model.cls_r.parameters())
        else :
            disturb_params = list(model.of_encoder.parameters())
            disturb_params_cls = list(model.cls_o.parameters())
        loss_list = []

        f_param_grads = torch.autograd.grad(loss_ori.mean(), disturb_params, retain_graph=True)
        f_param_grads_real = list(f_param_grads)
        grad_list = f_param_grads_real

        f_param_grads_cls = torch.autograd.grad(loss_ori.mean(), disturb_params_cls, retain_graph=True)
        f_param_grads_real_cls = list(f_param_grads_cls)
        grad_list_cls = f_param_grads_real_cls

        del f_param_grads, f_param_grads_real, f_param_grads_cls, f_param_grads_real_cls
        # ----- Calculate and add disturbances -----
        param_c = 0
        for param in disturb_params:
            grad_c = grad_list[param_c]
            if grad_c is not None:
                grad_c_norm = torch.norm(grad_c)
                rho_c = cfg['train']['noise_ratio']
                denominator = grad_c / grad_c_norm
                noise = rho_c * 1.0 * denominator
                param.data = param.data + noise
            param_c += 1

        param_c = 0
        for param in disturb_params_cls:
            grad_c = grad_list_cls[param_c]
            if grad_c is not None:
                grad_c_norm = torch.norm(grad_c)
                rho_c = cfg['train']['noise_ratio_cls']
                denominator = grad_c / grad_c_norm
                noise = rho_c * 1.0 * denominator
                param.data = param.data + noise
            param_c += 1

        if state =='train_depth':
            o_tmp = model.cls_d(fea)
        elif state == 'train_of' :
            o_tmp = model.cls_o(fea)
        else :
            o_tmp = model.cls_r(fea)
        loss_tmp = F.cross_entropy(o_tmp, y, reduction='none')
        loss_list.extend(loss_tmp)
        del grad_list, fea, disturb_params, grad_list_cls, disturb_params_cls

        loss_list = torch.stack(loss_list)
        loss = (1 - cfg['train']['flat_ratio']) * loss_ori.mean() + cfg['train']['flat_ratio'] * loss_list.mean()
        loss_flat = loss_list.mean().item()

        pred_q = F.softmax(o, dim=1).argmax(dim=1)
        f1 = f1_score(pred_q.cpu(), torch.argmax(y, dim=1).cpu(), average='macro')
        correct = torch.eq(pred_q.cpu(), torch.argmax(y, dim=1).cpu()).sum().item()
        acc = correct / y.shape[0]
        mAP = compute_mAP(F.softmax(o, dim=1), y)
        optimizer.zero_grad()
        loss.backward()
        # ----- Gradient modification -----
        modules_depth = [m for n, m in model.depth_encoder.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules_depth = modules_depth + [m for n, m in model.cls_d.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules_rgb = [m for n, m in model.rgb_encoder.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules_rgb = modules_rgb + [m for n, m in model.cls_r.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules_of = [m for n, m in model.of_encoder.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules_of = modules_of + [m for n, m in model.cls_o.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        if state == 'train_depth':
            modules_now = modules_depth
            modules_pre = modules_of
        elif state == 'train_rgb':
            modules_now = modules_rgb
            modules_pre = modules_depth
        else :
            modules_now = modules_of
            modules_pre = modules_rgb
        for m in modules_now:
            if isinstance(m,nn.Linear):
                for n in modules_pre:
                    if m.weight.shape == n.weight.shape:
                        m.weight.grad = m.weight.grad @ model.P_matrix[n.weight]
        optimizer.step()
        tl.add(loss.item())
        ta.add(acc)

        if step % cfg['print_inteval'] == 0:
            print((
                'Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f},Training mAP:{train_mAP:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1, train_mAP=mAP))
            logger.info((
                'Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f},Training mAP:{train_mAP:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1, train_mAP=mAP))

    loss_ave = tl.item()
    acc_ave = ta.item()

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Average Loss:{loss_ave:.3f}, Average Acc:{acc_ave:.2f}').format(epoch=epoch,
                                                                                             loss_ave=loss_ave,
                                                                                             acc_ave=acc_ave))

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Acc:{acc_ave:.2f}').format(
        epoch=epoch, loss_ave=loss_ave, acc_ave=acc_ave))

    return model


def train_withoutP(epoch, train_loader, model, optimizer, logger,state):
    model.train()

    tl = Averager()
    ta = Averager()

    for step,(rgb,of,depth,y) in enumerate(train_loader):
        rgb = rgb.float().cuda()
        of = of.cuda()
        depth = depth.cuda()
        y = y.cuda()
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        if state =='train_rbg':
            rgb, targets_a, targets_b, lam = mixup_data(rgb, y, config['train']['mixup_alpha'])
        elif state == 'train_of':
            of, targets_a, targets_b, lam = mixup_data(of, y, config['train']['mixup_alpha'])
        else :
            depth, targets_a, targets_b, lam = mixup_data(depth, y, config['train']['mixup_alpha'])

        input_list = []
        input_list.append(rgb), input_list.append(of), input_list.append(depth),input_list.append(state)
        fea, o = model(input_list)

        loss_ori = mixup_criterion(criterion, o, targets_a, targets_b, lam)
        # ----- SAM optimization -----

        if state =='train_depth':
            disturb_params = list(model.depth_encoder.parameters())
            disturb_params_cls = list(model.cls_d.parameters())
        elif state == 'train_rgb':
            disturb_params = list(model.rgb_encoder.parameters())
            disturb_params_cls = list(model.cls_r.parameters())
        else :
            disturb_params = list(model.of_encoder.parameters())
            disturb_params_cls = list(model.cls_o.parameters())
        loss_list = []

        f_param_grads = torch.autograd.grad(loss_ori.mean(), disturb_params, retain_graph=True)
        f_param_grads_real = list(f_param_grads)
        grad_list = f_param_grads_real

        f_param_grads_cls = torch.autograd.grad(loss_ori.mean(), disturb_params_cls, retain_graph=True)
        f_param_grads_real_cls = list(f_param_grads_cls)
        grad_list_cls = f_param_grads_real_cls

        del f_param_grads, f_param_grads_real, f_param_grads_cls, f_param_grads_real_cls
        # ----- Calculate and add disturbances -----

        param_c = 0
        for param in disturb_params:
            grad_c = grad_list[param_c]
            if grad_c is not None:
                grad_c_norm = torch.norm(grad_c)
                rho_c = cfg['train']['noise_ratio']
                denominator = grad_c / grad_c_norm
                noise = rho_c * 1.0 * denominator
                param.data = param.data + noise
            param_c += 1

        param_c = 0
        for param in disturb_params_cls:
            grad_c = grad_list_cls[param_c]
            if grad_c is not None:
                grad_c_norm = torch.norm(grad_c)
                rho_c = cfg['train']['noise_ratio_cls']
                denominator = grad_c / grad_c_norm
                noise = rho_c * 1.0 * denominator
                param.data = param.data + noise
            param_c += 1
        if state =='train_depth':
            o_tmp = model.cls_d(fea)
        elif state == 'train_of' :
            o_tmp = model.cls_o(fea)
        else :
            o_tmp = model.cls_r(fea)
        loss_tmp = F.cross_entropy(o_tmp, y, reduction='none')
        loss_list.extend(loss_tmp)
        del grad_list, fea, disturb_params, grad_list_cls, disturb_params_cls

        loss_list = torch.stack(loss_list)
        loss = (1 - cfg['train']['flat_ratio']) * loss_ori.mean() + cfg['train']['flat_ratio'] * loss_list.mean()
        loss_flat = loss_list.mean().item()
        pred_q = F.softmax(o, dim=1).argmax(dim=1)
        f1 = f1_score(pred_q.cpu(), torch.argmax(y, dim=1).cpu(), average='macro')
        correct = torch.eq(pred_q.cpu(), torch.argmax(y, dim=1).cpu()).sum().item()
        acc = correct / y.shape[0]
        mAP = compute_mAP(F.softmax(o, dim=1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        ta.add(acc)

        if step % cfg['print_inteval'] == 0:
            print((
                      'Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f},Training mAP:{train_mAP:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1, train_mAP=mAP))
            logger.info((
                            'Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f},Training mAP:{train_mAP:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1, train_mAP=mAP))

    loss_ave = tl.item()
    acc_ave = ta.item()

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Average Loss:{loss_ave:.3f}, Average Acc:{acc_ave:.2f}').format(epoch=epoch,
                                                                                             loss_ave=loss_ave,
                                                                                             acc_ave=acc_ave))

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Acc:{acc_ave:.2f}').format(
        epoch=epoch, loss_ave=loss_ave, acc_ave=acc_ave))

    return model


def val(epoch, val_loader, model, logger, state):
    model.eval()
    pred_list = []
    label_list = []
    soft_pred_list = []
    one_hot_label = []
    with torch.no_grad():
        for step, (rgb, of, depth, y) in enumerate(val_loader):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            rgb = rgb.float().cuda()
            of = of.cuda()
            depth = depth.cuda()
            input_list = []
            input_list.append(rgb), input_list.append(of), input_list.append(depth), input_list.append(state)
            o = model(input_list)
            soft_pred_list = soft_pred_list + o.tolist()
            pred_q = o.argmax(dim=1)
            pred_list = pred_list + pred_q.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        acc = correct / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred_list), torch.Tensor(one_hot_label))

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('State:{state} Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f}').format(epoch=epoch, f1=f1,
                                                                                            state=state, acc=acc,
                                                                                            mAP=mAP))

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('State:{state} Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f}').format(epoch=epoch, f1=f1,
                                                                                                  state=state, acc=acc,
                                                                                                  mAP=mAP))
    return f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str)

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = NvGestureDataset(config, mode='train')
    test_dataset = NvGestureDataset(config, mode='test')


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, sampler=None,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    # ----- MODEL -----

    best_acc =0
    state='train_depth'
    state_dict={'train_depth':'train_rgb','train_rgb':'train_of','train_of':'train_depth'}
    train_stage=0
    for train_stage_epoch in cfg['train']['epoch_list']:
        model = create_model(cfg).cuda()
        train_stage+=1
        if train_stage != 1:
            checkpoint=torch.load('best_model.pth')
            model.load_state_dict(checkpoint)
            with torch.no_grad():
                compute_P_matrix(model, train_loader, config, state)
            state = state_dict[state]
        else :
            state ='train_depth'
        lr_adjust = config['train']['optimizer']['lr']
        # ----- SET OR RESET OPTIMIZER -----
        if config['train']['optimizer']['type'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                                  momentum=config['train']['optimizer']['momentum'],
                                  weight_decay=config['train']['optimizer']['wc'])
        elif config['train']['optimizer']['type'] == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=lr_adjust, betas=(0.9, 0.99),
                                   weight_decay=config['train']['optimizer']['wc'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
        for epoch in range(train_stage_epoch):
            print(('Epoch {epoch:d} is pending...').format(epoch=epoch))
            logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

            if train_stage == 1:
                scheduler.step()
                model = train_withoutP(epoch, train_loader, model, optimizer, logger,state)
            else:
                scheduler.step()
                model = train_withP(epoch, train_loader, model, optimizer, logger,state)

            _ = val(epoch, test_loader, model, logger, 'test_depth')
            _ = val(epoch, test_loader, model, logger, 'test_rgb')
            _ = val(epoch, test_loader, model, logger, 'test_of')
            acc = val(epoch, test_loader, model, logger, 'test')
            if acc > best_acc:
                best_acc = acc
                print('Find a better model and save it!')
                logger.info('Find a better model and save it!')
                torch.save(model.state_dict(), 'best_model.pth')
        del model,optimizer,scheduler