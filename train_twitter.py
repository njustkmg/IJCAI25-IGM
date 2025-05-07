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
from sklearn.metrics import f1_score
from configs.template import config
from transformers import BertTokenizer
from datasets import create_dataset
from models.TextImage import create_model
from utils.loss import mixup_criterion


from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
    get_optimizer,
    get_scheduler,
    mixup_data
)
def compute_P_matrix(model, train_loader, config,state):
    model.eval()
    # ----- Calculate the feature variance -----
    if state =='train_text':
        modules = [m for n, m in model.cls_t.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
    else :
        modules = [m for n, m in model.cls_i.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
    handles = []
    for m in modules:
        handles.append(m.register_forward_hook(hook=model.compute_cov))
    tokenizer = BertTokenizer.from_pretrained('checkpoint/bert')

    for step, (image, text, y) in enumerate(train_loader):
        image = image.float().cuda()
        text_input = tokenizer(text, padding='longest', max_length=50, return_tensors="pt").to(image.device)
        input_list = []
        input_list.append(text_input), input_list.append(image), input_list.append(state)
        model.forward(input_list)
    # ----- Calculate the modification matrixs -----
    model.get_P_matrix(modules, model.fea_in, config)
    for h in handles:
        h.remove()
def train_withP(epoch, train_loader, model, optimizer, logger,state):
    model.train()

    tl = Averager()
    ta = Averager()
    tokenizer = BertTokenizer.from_pretrained('checkpoint/bert')
    for step, (image, text, y) in enumerate(train_loader):
        image = image.float().cuda()
        y = y.cuda()
        text_input = tokenizer(text, padding='longest', max_length=50, return_tensors="pt").to(image.device)
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        if state == 'train_image':
            image, targets_a, targets_b, lam = mixup_data(image, y, config['train']['mixup_alpha'])
        input_list = []
        input_list.append(text_input), input_list.append(image), input_list.append(state)
        fea, o = model(input_list)
        # ----- SAM optimization -----

        if state == 'train_text':
            loss_ori = F.cross_entropy(o, y, reduction='none')
            disturb_params = list(model.text_encoder.parameters())
            disturb_params_cls = list(model.cls_t.parameters())
        else :
            loss_ori = mixup_criterion(criterion, o, targets_a, targets_b, lam)
            disturb_params = list(model.visual_encoder.parameters())
            disturb_params_cls = list(model.cls_i.parameters())
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

        if state == 'train_text':
            o_tmp = model.cls_t(fea)
        else :
            o_tmp = model.cls_i(fea)
        loss_tmp = F.cross_entropy(o_tmp, y, reduction='none')
        loss_list.extend(loss_tmp)
        del grad_list, fea, disturb_params, grad_list_cls, disturb_params_cls

        loss_list = torch.stack(loss_list)
        loss = (1 - cfg['train']['flat_ratio']) * loss_ori.mean() + cfg['train']['flat_ratio'] * loss_list.mean()
        loss_flat = loss_list.mean().item()

        pred_q = F.softmax(o, dim=1).argmax(dim=1)
        f1=f1_score(pred_q.cpu(),y.cpu(),average='macro')
        correct=torch.eq(pred_q.cpu(),y.cpu()).sum().item()
        acc = correct / y.shape[0]
        optimizer.zero_grad()
        loss.backward()
        # ----- Gradient modification -----

        modules_text = [m for n, m in model.cls_t.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        modules_image = [m for n, m in model.cls_i.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        if state == 'train_text':
            modules_now = modules_text
            modules_pre = modules_image
        else :
            modules_now = modules_image
            modules_pre = modules_text

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
                'Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1))
            logger.info((
                'Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1))

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
    tokenizer = BertTokenizer.from_pretrained('checkpoint/bert')
    for step,(image, text, y) in enumerate(train_loader):
        image = image.cuda()
        y = y.cuda()
        text_input = tokenizer(text, padding='longest', max_length=50, return_tensors="pt").to(image.device)
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        if state =='train_image':
            image, targets_a, targets_b, lam = mixup_data(image, y, config['train']['mixup_alpha'])

        input_list = []
        input_list.append(text_input), input_list.append(image), input_list.append(state)
        fea, o = model(input_list)
        # ----- SAM optimization -----

        if state =='train_text':
            loss_ori = F.cross_entropy(o, y, reduction='none')
            disturb_params = list(model.text_encoder.parameters())
            disturb_params_cls = list(model.cls_t.parameters())
        else :
            loss_ori = mixup_criterion(criterion, o, targets_a, targets_b, lam)
            disturb_params = list(model.visual_encoder.parameters())
            disturb_params_cls = list(model.cls_i.parameters())
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
        if state =='train_text':
            o_tmp = model.cls_t(fea)
        else :
            o_tmp = model.cls_i(fea)
        loss_tmp = F.cross_entropy(o_tmp, y, reduction='none')
        loss_list.extend(loss_tmp)
        del grad_list, fea, disturb_params, grad_list_cls, disturb_params_cls

        loss_list = torch.stack(loss_list)
        loss = (1 - cfg['train']['flat_ratio']) * loss_ori.mean() + cfg['train']['flat_ratio'] * loss_list.mean()
        loss_flat = loss_list.mean().item()

        pred_q = F.softmax(o, dim=1).argmax(dim=1)
        f1=f1_score(pred_q.cpu(),y.cpu(),average='macro')
        correct=torch.eq(pred_q.cpu(),y.cpu()).sum().item()
        acc = correct / y.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        ta.add(acc)

        if step % cfg['print_inteval'] == 0:
            print(('Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1))
            logger.info(('Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f},Training F1:{train_f1:.2f}').format(
                train_loss=loss.item(), ori_loss=loss_ori.mean().item(), flat_loss=loss_flat, train_acc=acc,
                train_f1=f1))

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
    tokenizer = BertTokenizer.from_pretrained('checkpoint/bert')
    pred_list = []
    label_list = []
    with torch.no_grad():
        for step, (image, text, y) in enumerate(val_loader):
            label_list = label_list + y.tolist()
            image = image.cuda()
            y = y.cuda()
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)
            input_list = []
            input_list.append(text_input), input_list.append(image), input_list.append(state)
            o = model(input_list)

            pred_q = F.softmax(o, dim=1)
            pred_q = pred_q.argmax(dim=1)
            pred_list = pred_list + pred_q.tolist()


        f1 = f1_score(label_list, pred_list, average='macro')
        correct=np.sum(np.array(label_list)==np.array(pred_list))
        acc = correct / len(label_list)

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('State:{state} Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f}').format(epoch=epoch, f1=f1,state=state, acc=acc))
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('State:{state} Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f}').format(epoch=epoch, f1=f1,state=state, acc=acc))
    return acc


if __name__ == '__main__':

    # ----- LOAD PARAM -----
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
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset, test_dataset = create_dataset('twitter', config)


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                                  num_workers=cfg['train']['num_workers'], pin_memory=True, sampler=None,
                                  drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    # ----- MODEL -----
    best_acc = 0
    state='train_text'
    state_dict={'train_text':'train_image','train_image':'train_text'}
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
            state ='train_text'
        lr_adjust = config['train']['optimizer']['lr']
        # ----- SET OR RESET OPTIMIZER -----
        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer, cfg['train']['stage'])
        for epoch in range(train_stage_epoch):
            print(('Epoch {epoch:d} is pending...').format(epoch=epoch))
            logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

            if train_stage == 1:
                scheduler.step()
                model = train_withoutP(epoch, train_loader, model, optimizer, logger,state)
            else:
                scheduler.step()
                model = train_withP(epoch, train_loader, model, optimizer, logger,state)
            _ = val(epoch, test_loader, model, logger, 'test_text')
            _ = val(epoch, test_loader, model, logger, 'test_image')
            acc = val(epoch, test_loader, model, logger, 'test')
            if acc > best_acc:
                best_acc = acc
                print('Find a better model and save it!')
                logger.info('Find a better model and save it!')
                torch.save(model.state_dict(), 'best_model.pth')
        del model,optimizer,scheduler