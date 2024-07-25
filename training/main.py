# *** *** *** ***
# Boiler Codes - Import Dependencies

if __name__ == '__main__': # used for Windows freeze_support() issues, indent the rest of the lines below
    from torch.optim import lr_scheduler
    import torch.nn as nn
    import torch.optim as optim
    import torch
    import torch.multiprocessing
    import pickle
    import numpy as np
    import random
    import copy
    import os, sys, glob, shutil
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import json
    import argparse
    # from torchsummary import summary

    sys.path.insert(0, os.path.abspath('.'))
    from configs.params import *
    from configs import params
    from configs import datasets_config as config
    from data import data_loader as data_loader
    from network.logits import ArcFace
    import network.ael_net as net
    import train
    from eval import verification
    from eval import identification
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Imported.")

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--method', default=params.method, type=str,
                        help='method (backbone)')
    parser.add_argument('--remarks', default=params.remarks, type=str,
                        help='additional remarks')
    parser.add_argument('--write_log', default=params.write_log, type=bool,
                        help='flag to write logs')
    parser.add_argument('--dim', default=params.dim, type=int, metavar='N',
                        help='embedding dimension')
    parser.add_argument('--epochs', default=params.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=params.lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--w_decay', '--w_decay', default=params.w_decay, type=float,
                        metavar='Weight Decay', help='weight decay')
    parser.add_argument('--dropout', '--dropout', default=params.dropout, type=float,
                        metavar='Dropout', help='dropout probability')
    parser.add_argument('--pretrained', default='./models/pretrained/MobileFaceNet_1024.pt', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')

    args = parser.parse_args()

    # Determine if an nvidia GPU is available
    device = params.device
    start_ = datetime.now()
    start_string = start_.strftime("%Y%m%d_%H%M%S")

    # For reproducibility, Seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    file_main_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    print('Running on device: {}'.format(device))

    # *** *** *** ***
    # Load Dataset and Display Other Parameters

    # Face images
    face_train_dir = config.trainingdb['face_train']
    face_loader_train, face_train_set = data_loader.gen_data(face_train_dir, 'train_rand', type='face', aug='True')
    face_loader_train_tl, face_train_tl_set = data_loader.gen_data(face_train_dir, 'train', type='face', aug='True')

    # Periocular Images
    peri_train_dir = config.trainingdb['peri_train']
    peri_loader_train, peri_train_set = data_loader.gen_data(peri_train_dir, 'train_rand', type='periocular', aug='True')
    peri_loader_train_tl, peri_train_tl_set = data_loader.gen_data(peri_train_dir, 'train', type='periocular', aug='True')

    # Validation Periocular (Gallery/Val + Probe/Test)
    peri_val_dir = config.trainingdb['peri_val']
    peri_loader_val, peri_val_set = data_loader.gen_data(peri_val_dir, 'test', type='periocular', aug='False')
    peri_test_dir = config.trainingdb['peri_test']
    peri_loader_test, peri_test_set = data_loader.gen_data(peri_test_dir, 'test', type='periocular', aug='False')

    # Validation Face (Gallery/Val + Probe/Test)
    face_val_dir = config.trainingdb['face_val']
    face_loader_val, face_val_set = data_loader.gen_data(face_val_dir, 'test', type='face', aug='False')
    face_test_dir = config.trainingdb['face_test']
    face_loader_test, face_test_set = data_loader.gen_data(face_test_dir, 'test', type='face', aug='False')

    # Test Periocular (Ethnic)
    ethnic_peri_gallery_dir = config.ethnic['peri_gallery']
    ethnic_peri_probe_dir = config.ethnic['peri_probe']
    ethnic_peri_val_loader, ethnic_peri_val_set = data_loader.gen_data(ethnic_peri_gallery_dir, 'test', type='periocular')
    ethnic_peri_test_loader, ethnic_peri_test_set = data_loader.gen_data(ethnic_peri_probe_dir, 'test', type='periocular')

    # Test Face (Ethnic)
    ethnic_face_gallery_dir = config.ethnic['face_gallery']
    ethnic_face_probe_dir = config.ethnic['face_probe']
    ethnic_face_val_loader, ethnic_face_val_set = data_loader.gen_data(ethnic_face_gallery_dir, 'test', type='face')
    ethnic_face_test_loader, ethnic_face_test_set = data_loader.gen_data(ethnic_face_probe_dir, 'test', type='face')

    # Set and Display all relevant parameters
    print('\n***** Face ( Train ) *****\n')
    face_num_train = len(face_train_set)
    face_num_sub = len(face_train_set.classes)
    print(face_train_set)
    print('Num. of Sub.\t\t:', face_num_sub)
    print('Num. of Train. Imgs (Face) \t:', face_num_train)

    print('\n***** Periocular ( Train ) *****\n')
    peri_num_train = len(peri_train_set)
    peri_num_sub = len(peri_train_set.classes)
    print(peri_train_set)
    print('Num. of Sub.\t\t:', peri_num_sub)
    print('Num. of Train Imgs (Periocular) \t:', peri_num_train)

    print('\n***** Periocular ( Validation (Gallery) ) *****\n')
    peri_num_val = len(peri_val_set)
    print(peri_val_set)
    print('Num. of Sub.\t\t:', len(peri_val_set.classes))
    print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    print('\n***** Periocular ( Validation (Probe) ) *****\n')
    peri_num_test = len(peri_test_set)
    print(peri_test_set)
    print('Num. of Sub.\t\t:', len(peri_test_set.classes))
    print('Num. of Test Imgs (Periocular) \t:', peri_num_test)

    # print('\n***** Face ( Validation (Gallery) ) *****\n')
    # peri_num_val = len(face_val_set)
    # print(face_val_set)
    # print('Num. of Sub.\t\t:', len(face_val_set.classes))
    # print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    # print('\n***** Face ( Test (Probe) ) *****\n')
    # face_num_test = len(face_test_set)
    # print(face_test_set)
    # print('Num. of Sub.\t\t:', len(face_test_set.classes))
    # print('Num. of Test Imgs (Periocular) \t:', face_num_test)

    print('\n***** Other Parameters *****\n')
    print('Start Time \t\t: ', start_string)
    print('Method (Backbone)\t: ', args.method)
    print('Remarks\t\t\t: ', args.remarks)
    print('Net. Descr.\t\t: ', net_descr)
    print('Seed\t\t\t: ', seed)
    print('Batch # Sub.\t\t: ', batch_sub)
    print('Batch # Samp.\t\t: ', batch_samp)
    print('Batch Size\t\t: ', batch_size)
    print('Random Batch Size\t\t: ', random_batch_size)
    print('Test Batch Size\t\t: ', test_batch_size)
    print('Emb. Dimension\t\t: ', args.dim)
    print('# Epoch\t\t\t: ', epochs)
    print('Learning Rate\t\t: ', args.lr)
    print('LR Scheduler\t\t: ', lr_sch)
    print('Weight Decay\t\t: ', args.w_decay)
    print('Dropout Prob.\t\t: ', args.dropout)
    print('BN Flag\t\t\t: ', bn_flag)
    print('BN Momentum\t\t: ', bn_moment)
    print('Scaling\t\t\t: ', af_s)
    print('Margin\t\t\t: ', af_m)
    print('Save Flag\t\t: ', save)
    print('Log Writing\t\t: ', args.write_log)


    # *** *** *** ***
    # Load Pre-trained Model, Define Loss and Other Hyperparameters for Training

    print('\n***** *****\n')
    print('Loading Pretrained Model' )  
    print()

    train_mode = 'eval'
    model = net.AEL_Net(embedding_size = args.dim, do_prob = args.dropout).eval().to(device)
    # model = net.Resnet(num_layers=50, feat_dim = args.dim, drop_ratio = args.dropout).eval().to(device)
    # model = net.MobileFaceNet(embedding_size= args.dim, do_prob=args.dropout).eval().to(device)

    load_model_path = args.pretrained
    state_dict_loaded = model.state_dict()
    # state_dict_pretrained = torch.load(load_model_path, map_location = device)
    state_dict_pretrained = torch.load(load_model_path, map_location = device)['state_dict']
    state_dict_temp = {}

    for k in state_dict_loaded:
        if 'encoder' not in k:
            # state_dict_temp[k] = state_dict_pretrained[k]
            state_dict_temp[k] = state_dict_pretrained['backbone.'+k]
        else:
            print(k, 'not loaded!')
    state_dict_loaded.update(state_dict_temp)
    model.load_state_dict(state_dict_loaded)
    del state_dict_loaded, state_dict_pretrained, state_dict_temp

    # for multiple GPU usage, set device in params to torch.device('cuda') without specifying GPU ID.
    # model = torch.nn.DataParallel(model).cuda()
    ####

    out_features = args.dim 

    # for AELNet or MobileFaceNet
    in_features  = model.linear.in_features
    # model.linear = nn.Linear(in_features, out_features, bias = True)                      # Deep Embedding Layer
    # model.bn = nn.BatchNorm1d(out_features, eps = 1e-5, momentum = 0.1, affine = True) # BatchNorm1d Layer

    #### model summary
    # torch.cuda.empty_cache()
    # import os
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # summary(model.to(device),(3,112,112))

    # *** ***

    # print('\n***** *****\n')
    print('Appending FC to model ...' )  
    psi_1 = ArcFace(in_features = out_features, out_features = face_num_sub, s = 128.0, m = af_m, device = device).eval().to(device)
    psi_2 = ArcFace(in_features = out_features, out_features = face_num_sub, s = af_s, m = af_m, device = device).eval().to(device)
    psi_3 = ArcFace(in_features = out_features, out_features = face_num_sub, s = af_s, m = af_m, device = device).eval().to(device)
    psi_4 = ArcFace(in_features = out_features, out_features = face_num_sub, s = af_s, m = af_m, device = device).eval().to(device)

    # **********

    print('Re-Configuring Model with Psi 1 to 4 ... ' ) 
    print()

    # *** ***
    # model : Determine parameters to be freezed, or unfreezed
    for name, param in model.named_parameters():
        # param.requires_grad = True
        if epochs_pre > 0:
            param.requires_grad = False
            if name in ['linear.weight', 'linear.bias', 'bn.weight', 'bn.bias'] or 'encoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True
        

    # model : Display all learnable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('model (requires grad)\t:', name)

    # print('model\t: ALL Parameters')
            
    # *** 

    # model : Freeze or unfreeze BN parameters
    for name, layer in model.named_modules():
        if isinstance(layer,torch.nn.BatchNorm2d): #or isinstance(layer,torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.LayerNorm):
            # ***
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True
            # *** 

    if bn_flag == -1:
        print('model\t: EXCLUDE BatchNorm2D Parameters')
    elif bn_flag == 0 or bn_flag == 1:
        print('model\t: INCLUDE BatchNorm2D.weight & bias')

    # *** ***

    # FC Layers : Determine parameters to be freezed, or unfreezed
    for param in psi_1.parameters():
        param.requires_grad = True

    for param in psi_2.parameters():
        param.requires_grad = True

    for param in psi_3.parameters():
        param.requires_grad = True

    for param in psi_4.parameters():
        param.requires_grad = True

    # ********** 
    # Set an optimizer, scheduler, etc.
    loss_fn = { 'loss_ce' : torch.nn.CrossEntropyLoss()}
            
    parameters_backbone = [p for p in model.parameters() if p.requires_grad]
    parameters_psi_1 = [p for p in psi_1.parameters() if p.requires_grad]
    parameters_psi_2 = [p for p in psi_2.parameters() if p.requires_grad]
    parameters_psi_3 = [p for p in psi_3.parameters() if p.requires_grad]
    parameters_psi_4 = [p for p in psi_4.parameters() if p.requires_grad]

    optimizer = optim.AdamW([   {'params': parameters_backbone},
                                {'params': parameters_psi_1, 'lr': lr*10, 'weight_decay': args.w_decay},
                                {'params': parameters_psi_2, 'lr': lr*10, 'weight_decay': args.w_decay},
                                {'params': parameters_psi_3, 'lr': lr*10, 'weight_decay': args.w_decay},
                                {'params': parameters_psi_4, 'lr': lr*10, 'weight_decay': args.w_decay},
                            ], lr = args.lr, weight_decay = args.w_decay)

    # optimizer = optim.AdamW(opt_params, lr = args.lr, weight_decay = args.w_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_sch, gamma = 0.1)

    metrics = { 'fps': train.BatchTimer(), 'acc': train.accuracy}

    net_params = { 'network' : net_descr, 'method' : args.method, 'remarks' : args.remarks, 'epochs' : epochs, 'epochs_pre' : epochs_pre, 
                'face_num_sub' : face_num_sub, 'peri_num_sub': peri_num_sub, 'scale' : af_s, 'margin' : af_m,
                'weight_1' : weight_1, 'weight_2' : weight_2,
                'lr' : args.lr, 'lr_sch': lr_sch, 'w_decay' : args.w_decay, 'dropout' : args.dropout,
                'batch_sub' : batch_sub, 'batch_samp' : batch_samp, 'batch_size' : batch_size, 'dims' : args.dim, 'seed' : seed }


    # *** *** *** ***
    #### Model Training

    #### Define Logging
    train_mode = 'train'    
    log_folder = "./logs/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks)
    if not os.path.exists(log_folder) and args.write_log is True:
        os.makedirs(log_folder)
    log_nm = log_folder + "/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks) + ".txt"

    # Files Backup
    if args.write_log is True: # only backup if there is log
        # copy main and training files as backup
        for files in glob.glob(os.path.join(file_main_path, '*')):
            if '__' not in files: # ignore __pycache__
                shutil.copy(files, log_folder)
                print(files)
        # networks and logits
        py_extension = '.py'
        desc = file_main_path.split('/')[-1]
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'ael_net' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'logits' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'configs'), 'params' + py_extension), log_folder)

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write(str(net_descr) + "\n")
        file.write('Training started at ' + str(start_) + ".\n\n")
        file.write('Model parameters: \n')
        file.write(json.dumps(net_params) + "\n\n")
        file.close()


    # *** *** *** ***
    #### Begin Training

    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0

    peri_best_test_acc = 0
    peri_best_pr_test_acc = 0

    best_model = copy.deepcopy(model.state_dict())
    best_psi_1 = copy.deepcopy(psi_1.state_dict())
    best_psi_2 = copy.deepcopy(psi_2.state_dict())
    best_psi_3 = copy.deepcopy(psi_3.state_dict())
    best_psi_4 = copy.deepcopy(psi_4.state_dict())

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    model.eval().to(device)
    psi_1.eval().to(device)
    psi_2.eval().to(device)
    psi_3.eval().to(device)
    psi_4.eval().to(device)    

    for epoch in range(epochs):    
        print()
        print()        
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train().to(device)
        psi_1.train().to(device)    
        psi_2.train().to(device)
        psi_3.train().to(device)
        psi_4.train().to(device)

        if epoch + 1 > epochs_pre:
            for name, param in model.named_parameters():
                param.requires_grad = True
        
        # Use running_stats for training and testing
        if bn_flag != 2:
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                    # or \
                    # isinstance(layer,torch.nn.modules.batchnorm.BatchNorm1d) or \
                    # isinstance(layer,torch.nn.modules.LayerNorm):                
                    layer.eval()
                    layer.weight.requires_grad = True
                    layer.bias.requires_grad = True
                
        train_acc, loss = train.run_train(model, psi_1 = psi_1, psi_2 = psi_2, psi_3 = psi_3, psi_4 = psi_4,
                                            face_loader = face_loader_train, peri_loader = peri_loader_train, face_loader_tl = face_loader_train_tl, peri_loader_tl = peri_loader_train_tl,
                                            net_params = net_params, loss_fn = loss_fn, optimizer = optimizer, 
                                            scheduler = scheduler, batch_metrics = metrics, 
                                            show_running = True, device = device, writer = writer)   
        print('Loss : ', loss)
        # *** ***    
        model.eval().to(device)
        psi_1.eval().to(device)
        psi_2.eval().to(device)
        psi_3.eval().to(device)
        psi_4.eval().to(device)
        # *****
        
        # print('Periocular')
        peri_val_acc = identification.crossmodal_id(model, peri_loader_val, peri_loader_test, device=device, peri_flag=True, gallery='peri')
        peri_val_acc = np.around(peri_val_acc, 4)
        print('Validation Rank-1 IR (Cross - Periocular)\t: ', peri_val_acc)    

        # Testing (Ethnic)
        ethnic_cross_peri_acc = identification.crossmodal_id(model, ethnic_face_test_loader, ethnic_peri_val_loader,
                                                            device=device, face_model=None, peri_model=None, gallery='peri')
        ethnic_cross_peri_acc = np.around(ethnic_cross_peri_acc, 4)
        print('Test Rank-1 IR (Ethnic Cross - Periocular)\t: ', ethnic_cross_peri_acc)   

        ethnic_cross_face_acc = identification.crossmodal_id(model, ethnic_face_val_loader, ethnic_peri_test_loader, 
                                                            device=device, face_model=None, peri_model=None, gallery='face')
        ethnic_cross_face_acc = np.around(ethnic_cross_face_acc, 4)
        print('Test Rank-1 IR (Ethnic Cross - Face)\t: ', ethnic_cross_face_acc)    
        
        if args.write_log is True:
            file = open(log_nm, 'a+')
            file.write(str('Epoch {}/{}'.format(epoch + 1, epochs)) + "\n")
            file.write('Loss : ' + str(loss) + "\n")
            file.write('Validation Rank-1 IR (Periocular)\t: ' + str(peri_val_acc) + "\n")
            file.write('Test Rank-1 IR (Cross Periocular) \t: ' + str(ethnic_cross_peri_acc) + "\n")
            file.write('Test Rank-1 IR (Cross Face) \t: ' + str(ethnic_cross_face_acc) + "\n\n")
            file.close()

        # save best model based on Rank-1 IR validation
        if peri_val_acc >= peri_best_test_acc and epoch + 1 >= lr_sch[0] and save == True:         
            best_epoch = epoch + 1
            best_train_acc = train_acc
            peri_best_test_acc = peri_val_acc

            best_model = copy.deepcopy(model.state_dict())
            best_psi_1 = copy.deepcopy(psi_1.state_dict())
            best_psi_2 = copy.deepcopy(psi_2.state_dict())
            best_psi_3 = copy.deepcopy(psi_3.state_dict())
            best_psi_4 = copy.deepcopy(psi_4.state_dict())

            print('\n***** *****\n')
            print('Saving Best Model & Rank-1 IR ... ')
            print()
            
            # Set save_best_model_path
            tag = str(args.method) +  '/' + net_tag + '_' + str(batch_sub) + '_' + str(batch_samp) + '/'
            
            save_best_model_dir = './models/best_model/' + tag
            if not os.path.exists(save_best_model_dir):
                os.makedirs(save_best_model_dir)

            save_best_psi_1_dir = './models/best_psi_1/' + tag
            if not os.path.exists(save_best_psi_1_dir):
                os.makedirs(save_best_psi_1_dir)
                
            save_best_psi_2_dir = './models/best_psi_2/' + tag
            if not os.path.exists(save_best_psi_2_dir):
                os.makedirs(save_best_psi_2_dir)
            
            save_best_psi_3_dir = './models/best_psi_3/' + tag
            if not os.path.exists(save_best_psi_3_dir):
                os.makedirs(save_best_psi_3_dir)
            
            save_best_psi_4_dir = './models/best_psi_4/' + tag
            if not os.path.exists(save_best_psi_4_dir):
                os.makedirs(save_best_psi_4_dir)
                
            save_best_acc_dir = './models/best_acc/' + tag
            if not os.path.exists(save_best_acc_dir):
                os.makedirs(save_best_acc_dir)  
                    
            tag = str(args.method) + '_' + str(args.remarks)    

            save_best_model_path = save_best_model_dir + tag + '_' + str(start_string) + '.pth'
            save_best_psi_1_path = save_best_psi_1_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_psi_2_path = save_best_psi_2_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_psi_3_path = save_best_psi_3_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_psi_4_path = save_best_psi_4_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_acc_path = save_best_acc_dir + tag + '_' + str(start_string) + '.pkl' 
                    
            print('Best Model Pth\t: ', save_best_model_path)
            print('Best Psi-1 Pth\t: ', save_best_psi_1_path)
            print('Best Psi-2 Pth\t: ', save_best_psi_2_path)
            print('Best Psi-3 Pth\t: ', save_best_psi_3_path)
            print('Best Psi-4 Pth\t: ', save_best_psi_4_path)
            print('Best Rank-1 IR Pth\t: ', save_best_acc_path)

            # *** ***
            
            torch.save(best_model, save_best_model_path)
            torch.save(best_psi_1, save_best_psi_1_path)
            torch.save(best_psi_2, save_best_psi_2_path)
            torch.save(best_psi_3, save_best_psi_3_path)
            torch.save(best_psi_4, save_best_psi_4_path)

            with open(save_best_acc_path, 'wb') as f:
                pickle.dump([ best_epoch, best_train_acc, peri_best_test_acc, peri_best_pr_test_acc ], f)

    if args.write_log is True:
        file = open(log_nm, 'a+')
        end_ = datetime.now()
        end_string = end_.strftime("%Y%m%d_%H%M%S")
        file.write('Training completed at ' + str(end_) + ".\n\n")
        file.write("Model: " + str(save_best_model_path) + "\n\n")    
        file.close()


    # *** *** *** ***
    #### Identification and Verification for Test Datasets ( Ethnic, Pubfig, FaceScrub, IMDb Wiki, AR)

    print('\n**** Testing Evaluation (All Datasets) **** \n')
    #### Cross-modal Identification and Verification (Face and Periocular)
    print("Cross-Modal Rank-1 IR\n")
    cm_id_dict_p, cm_id_dict_f = identification.cm_id_main(model, root_pth=config.evaluation['identification'], face_model=None, peri_model=None, device=device)
    cm_id_dict_p, cm_id_dict_f = identification.get_avg(cm_id_dict_p), identification.get_avg(cm_id_dict_f)
    cm_id_dict_p = copy.deepcopy(cm_id_dict_p)
    cm_id_dict_f = copy.deepcopy(cm_id_dict_f)
    print('Average IR (Cross-Modal Periocular): \n', cm_id_dict_p['avg'], '±', cm_id_dict_p['std'])
    print('Average IR (Cross-Modal Face): \n', cm_id_dict_f['avg'], '±', cm_id_dict_f['std'])
    # *** #
    print("Cross-Modal EER\n")
    cm_eer_dict = verification.cm_verify(model, face_model=None, peri_model=None, emb_size=args.dim, root_drt=config.evaluation['verification'], device=device)
    cm_eer_dict = copy.deepcopy(cm_eer_dict)
    cm_eer_dict = verification.get_avg(cm_eer_dict)
    print('Average EER (Cross-Modal):', cm_eer_dict['avg'], '±', cm_eer_dict['std'])


    # *** *** *** ***
    # Dataset Performance Summary
    print('**** Testing Summary Results (All Datasets) **** \n')

    # *** ***
    print('\n\n Ethnic \n')
 
    ethnic_cm_ir_p = cm_id_dict_p['ethnic']
    ethnic_cm_ir_f = cm_id_dict_f['ethnic']
    ethnic_cm_eer = cm_eer_dict['ethnic']

    print('Cross-Modal Rank-1 IR - Periocular Gallery\t: ', ethnic_cm_ir_p)
    print('Cross-Modal Rank-1 IR - Face Gallery\t: ', ethnic_cm_ir_f)
    print('Cross-Modal EER \t: ', ethnic_cm_eer)


    # *** ***
    print('\n\n Pubfig \n')

    pubfig_cm_ir_p = cm_id_dict_p['pubfig']
    pubfig_cm_ir_f = cm_id_dict_f['pubfig']
    pubfig_cm_eer = cm_eer_dict['pubfig']
  
    print('Cross-Modal Rank-1 IR - Periocular Gallery\t: ', pubfig_cm_ir_p)
    print('Cross-Modal Rank-1 IR - Face Gallery\t: ', pubfig_cm_ir_f)
    print('Cross-Modal EER \t: ', pubfig_cm_eer)


    # *** ***
    print('\n\n FaceScrub\n')

    facescrub_cm_ir_p = cm_id_dict_p['facescrub']
    facescrub_cm_ir_f = cm_id_dict_f['facescrub']
    facescrub_cm_eer = cm_eer_dict['facescrub']

    print('Cross-Modal Rank-1 IR - Periocular Gallery\t: ', facescrub_cm_ir_p)
    print('Cross-Modal Rank-1 IR - Face Gallery\t: ', facescrub_cm_ir_f)
    print('Cross-Modal EER \t: ', facescrub_cm_eer)


    # *** *** *** ***
    print('\n\n IMDB Wiki \n')
   
    imdb_wiki_cm_ir_p = cm_id_dict_p['imdb_wiki']
    imdb_wiki_cm_ir_f = cm_id_dict_f['imdb_wiki']
    imdb_wiki_cm_eer = cm_eer_dict['imdb_wiki']
   
    print('Cross-Modal Rank-1 IR - Periocular Gallery\t: ', imdb_wiki_cm_ir_p)
    print('Cross-Modal Rank-1 IR - Face Gallery\t: ', imdb_wiki_cm_ir_f)
    print('Cross-Modal EER \t: ', imdb_wiki_cm_eer)


    # *** *** *** ***
    print('\n\n AR \n')
   
    ar_cm_ir_p = cm_id_dict_p['ar']
    ar_cm_ir_f = cm_id_dict_f['ar']
    ar_cm_eer = cm_eer_dict['ar']

    print('Cross-Modal Rank-1 IR - Periocular Gallery\t: ', ar_cm_ir_p)
    print('Cross-Modal Rank-1 IR - Face Gallery\t: ', ar_cm_ir_f)
    print('Cross-Modal EER \t: ', ar_cm_eer)

    # *** *** *** ***
    #### Average of all Datasets
    print('\n\n\n Calculating Average \n')

    avg_cm_p_ir = identification.get_avg(cm_id_dict_p)
    avg_cm_f_ir = identification.get_avg(cm_id_dict_f)
    avg_cm_eer = verification.get_avg(cm_eer_dict)

    print('Cross-modal Rank-1 IR - Periocular Gallery\t: ', avg_cm_p_ir['avg'], '±', avg_cm_p_ir['std'])
    print('Cross-modal Rank-1 IR - Face Gallery\t: ', avg_cm_f_ir['avg'], '±', avg_cm_f_ir['std'])
    print('Cross-modal EER \t: ', avg_cm_eer['avg'], '±', avg_cm_eer['std'])


    # *** *** *** ***
    # Write Final Performance Summaries to Log 

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write('****Ethnic:****')
        file.write('\nFinal Rank-1 IR (Cross-Modal): \n Periocular Gallery - ' + str(cm_id_dict_p['ethnic']) + ', \n Face Gallery - ' + str(cm_id_dict_f['ethnic'])+ '\n\n')   
        file.write('\nFinal EER (Cross-Modal): ' + str(cm_eer_dict['ethnic'])+ '\n\n')
        file.write('****Pubfig:****')
        file.write('\nFinal Rank-1 IR (Cross-Modal): \n Periocular Gallery - ' + str(cm_id_dict_p['pubfig']) + ', \n Face Gallery - ' + str(cm_id_dict_f['pubfig'])+ '\n\n')     
        file.write('\nFinal EER (Cross-Modal): ' + str(cm_eer_dict['pubfig'])+ '\n\n')
        file.write('****FaceScrub:****')
        file.write('\nFinal Rank-1 IR (Cross-Modal): \n Periocular Gallery - ' + str(cm_id_dict_p['facescrub']) + ', \n Face Gallery - ' + str(cm_id_dict_f['facescrub'])+ '\n\n')     
        file.write('\nFinal EER (Cross-Modal): ' + str(cm_eer_dict['facescrub'])+ '\n\n')
        file.write('****IMDB Wiki:****')
        file.write('\nFinal Rank-1 IR (Cross-Modal): \n Periocular Gallery - ' + str(cm_id_dict_p['imdb_wiki']) + ', \n Face Gallery - ' + str(cm_id_dict_f['imdb_wiki'])+ '\n\n')
        file.write('\nFinal EER (Cross-Modal): ' + str(cm_eer_dict['imdb_wiki'])+ '\n\n')
        file.write('****AR:****')
        file.write('\nFinal Rank-1 IR (Cross-Modal): \n Periocular Gallery - ' + str(cm_id_dict_p['ar']) + ', \n Face Gallery - ' + str(cm_id_dict_f['ar'])+ '\n\n')
        file.write('\nFinal EER (Cross-Modal): ' + str(cm_eer_dict['ar'])+ '\n\n') 


        file.write('\n\n **** Average **** \n\n')
        file.write('\nFinal Rank-1 IR (Cross-Modal): \n Periocular Gallery - ' + str(avg_cm_p_ir['avg']) + ' ± ' + str(avg_cm_p_ir['std'])  \
                   + ', \n Face Gallery - ' + str(avg_cm_f_ir['avg']) + ' ± ' + str(avg_cm_f_ir['std'])  + '\n\n')
        file.write('\nFinal EER (Cross-Modal): ' + str(avg_cm_eer['avg']) + ' ± ' + str(avg_cm_eer['std'])  + '\n\n')
        file.close()

    # *** ***