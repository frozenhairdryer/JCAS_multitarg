from aifc import Error
from imports import *
from functions import *
from NN_classes import *


def train_network(M=4,sigma_n=0.1,sigma_c=1,sigma_s=1,train_params=[50,100,0.01,1],w_r = 0.1,max_target=1,stage=None,NNs=None, plotting=True, setbehaviour="none", namespace="",loss_beam=0):
    """ 
    Training process of NN autoencoder

    M : number of constellation symbols
    sigma_n : noise standard deviation 
    train_params=[num_epochs,batches_per_epoch, learn_rate]
    w_r : Impact of the radar receiver; impact of communicationn is (1-w_r)
    stage : NNs should be trained serially; therefore there are 3 training stages:
        stage = 1: training of encoder, decoder, beamformer and angle estimation
        stage = 2: training of encoder, decoder, beamformer and estimation of angle uncertainty
        stage = 3: training of encoder, decoder, beamformer and target detection 
    
    M, sigma_n, modradius are lists of the same size 
    setbehaviour (str) : encompasses methods to take care of permutations;
        "setloss" -> use hausdorff distance for loss; permute in validation set
        "permute"
        "sortall"
        "sortinput"
        "none" : do nothing
        "ESPRIT" -> use esprit algorithm instead of NN 
    plotting (bool): toggles Plotting of PDFs into folder /figures 
    """
    encoding = ['counting', 'onehot'][0]
    benchmark = 1 
    canc = 0

    if M.size()!=sigma_n.size():
        raise error("M, sigma_n, need to be of same size (float)!")
    
    num_epochs=int(train_params[0])
    batches_per_epoch=int(train_params[1])
    cpr = int(train_params[3]) # communication per radar: integer of how many communication symbols are transmitted for the same radar estimation
    learn_rate =train_params[2]
    N_valid = 10000

    logging.info("Running training in training_routine_multitarget_nofft.py")
    logging.info("Maximum target number is %i" % max_target)
    logging.info("Set behaviour is %s" % setbehaviour )
    logging.info("loss_beam is %s" % str(loss_beam))
    #channel parameters
    lambda_txr = 0.1 # wavelength, in these simulations so far not used, as d=lambda_txr/2

    
    printing=False #suppresses all printed output but BMI
    
    # Generate Validation Data
    y_valid = torch.zeros(N_valid,dtype=int, device=device).to(device)
    y_valid= torch.randint(0,int(M),(N_valid*cpr,)).to(device)

    if plotting==True:
        # meshgrid for plotting
        ext_max = 2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = np.meshgrid(np.linspace(-ext_max,ext_max,200), np.linspace(-ext_max,ext_max,200))
        meshgrid = np.column_stack((np.reshape(mgx,(-1,1)),np.reshape(mgy,(-1,1))))
    
    if NNs==None:
        enc = Encoder(M).to(device)
        dec = Decoder(M).to(device)
        beam = Beamformer(kx=16,ky=1).to(device) # Set beamformer antenna number
        rad_rec = Radar_receiver(kx=16,ky=1,max_target=max_target, encoding=encoding).to(device)
    else:
        enc = NNs[0]
        dec = NNs[1]
        beam = NNs[2]
        rad_rec = NNs[3]
        encoding = rad_rec.encoding
        
    
    # Adam Optimizer
    # List of optimizers in case we want different learn-rates
    optimizer=[]
    optimizer.append(optim.Adam(enc.parameters(), lr=float(learn_rate)))
    optimizer.append(optim.Adam(dec.parameters(), lr=float(learn_rate)))
    

    if stage==1: # angle est
        optimizer.append(optim.Adam(rad_rec.rad_angle_est.parameters(), lr=float(learn_rate*10)))
        optimizer.append(optim.Adam(rad_rec.rad_detect.parameters(), lr=float(learn_rate)))
        optimizer.append(optim.Adam(beam.parameters(), lr=float(learn_rate)))
    elif stage==3: # target detection
        optimizer.append(optim.Adam(rad_rec.rad_detect.parameters(), lr=float(learn_rate)))
        optimizer.append(optim.Adam(rad_rec.rad_angle_est.parameters(), lr=float(learn_rate)))
        optimizer.append(optim.Adam(beam.parameters(), lr=float(learn_rate)))
    else:
        optimizer.append(optim.Adam(rad_rec.rad_angle_est.parameters(), lr=float(learn_rate*10)))
        optimizer.append(optim.Adam(rad_rec.rad_detect.parameters(), lr=float(learn_rate)))
        optimizer.append(optim.Adam(beam.parameters(), lr=float(learn_rate)))

    softmax = nn.Softmax(dim=1).to(device)

    # Cross Entropy loss
    ce_loss = nn.CrossEntropyLoss() # for one-hot encoding of targets
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss =nn.MSELoss()

    # fixed batch size of 10000
    batch_size_per_epoch = np.zeros(num_epochs, dtype=int)+10000

    validation_BER = []
    validation_SERs = torch.zeros((int(num_epochs)))
    validation_received = []
    det_error=[[],[]]
    sent_all=[]
    rmse_angle=[]
    rmse_benchmark=[]
    rmse_benchmark_2=[]
    d_angle_uncertain=[]
    CRB_azimuth = 100 # init high value for CRBs
    CRB_elevation = 100
    print('Start Training stage '+str(stage))
    logging.info("Start Training stage %s" % stage)

    bitnumber = int(torch.sum(torch.log2(M)))
    #BMI = torch.zeros(int(num_epochs),device=device)
    BMI_exact = torch.zeros((int(num_epochs), bitnumber), device=device)
    SNR = np.zeros(int(num_epochs))
    k = torch.arange(beam.kx)+1
    hausdorff = []
    #torch.autograd.set_detect_anomaly(True)

    for epoch in range(int(num_epochs)):
        for l_target in range(max_target):
            batch_labels = torch.empty(int(batch_size_per_epoch[epoch]*cpr),dtype=torch.long, device=device)
            validation_BER.append([])
            for step in range(int(batches_per_epoch)):
                # Generate training data: In most cases, you have a dataset and do not generate a training dataset during training loop
                # sample new mini-batch directory on the GPU (if available)
                decoded=torch.zeros(int(batch_size_per_epoch[epoch]*cpr),(torch.max(M)), device=device)
                batch_labels.random_(int(M))
                batch_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]*cpr), int(M), device=device)
                batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels.long()]=1

                theta_valid = torch.zeros((int(batch_size_per_epoch[epoch]),2), device=device)
                theta_valid[:,0] = np.pi/180*(torch.rand((int(batch_size_per_epoch[epoch])))*20+30)
                theta_valid[:,1] = np.pi/2#/180*(torch.rand((int(batch_size_per_epoch[epoch])))*1+90) # between 90 and 100 deg
                theta_valid = theta_valid.repeat(1,1,cpr).reshape((int(cpr*batch_size_per_epoch[epoch]),2))

                target_labels = torch.randint(l_target+2,(int(batch_size_per_epoch[epoch]),)).to(device) # Train in each epoch first 1 target, then 2, then ...
                ##encoding: [1,1,0,...] means 2 targets are detected

                target = torch.zeros((int(batch_size_per_epoch[epoch]),max_target)).to(device)
                label_tensor = torch.zeros(max_target+1,max_target).to(device)
                for x in range(max_target+1):
                    label_tensor[x] = torch.concat((torch.ones(x), torch.zeros(max_target-x)))
                target += label_tensor[target_labels] 
                target_onehot = torch.zeros((int(batch_size_per_epoch[epoch]),max_target+1)).to(device)
                target_onehot[np.arange(int(batch_size_per_epoch[epoch])),target_labels] = 1

                phi_valid = torch.zeros((int(batch_size_per_epoch[epoch]),2,max_target)).to(device)
                phi_valid[:,0,:] = np.pi/180*(torch.rand((int(batch_size_per_epoch[epoch]),max_target))*40-20) # paper: [-20 deg, 20 deg]
                phi_valid[:,1,:] = np.pi/2
                phi_valid[:,0,:] *= target
                phi_valid[:,1,:] *= target

                if setbehaviour=="sortall" or setbehaviour=="sortinput":
                    for l in range(int(batch_size_per_epoch[epoch])):
                        idx1 = torch.nonzero(phi_valid[l,0,:]).to(device)
                        idx0 = torch.nonzero(phi_valid[l,0,:] == 0).to(device)
                        if idx1.numel() > 0:
                            idx = torch.argsort(phi_valid[l,0,torch.squeeze(idx1,1)], descending=True).to(device)
                            if idx0.numel() > 0:
                                idx = torch.squeeze(torch.cat((idx,torch.squeeze(idx0,1))))
                            else:
                                idx = torch.squeeze(idx)
                        else:
                            idx = torch.squeeze(idx0).to(device)
                        phi_valid[l,0,:] = phi_valid[l,0,idx]
                        phi_valid[l,1,:] = phi_valid[l,1,idx]

                phi_valid_ex = phi_valid.repeat(1,cpr,1).reshape((cpr*batch_size_per_epoch[epoch],2,max_target))
                ### Start propagation through system
                # Use Beamforming
                direction = beam(np.pi/180*(torch.tensor([-20.0,20.0,30.0,50.0,0], device=device))).to(device) # give the angles in which targets/receivers are to be expected
                
                if direction.isnan().any():
                    raise Error("NaN encountered while training.")

                # Propagate (training) data through transmitter
                encoded = torch.unsqueeze(enc(batch_labels_onehot),1)
                
                modulated = torch.matmul(encoded, torch.unsqueeze(direction,0))# Apply Beamforming 

                # Propagate through channel
                t = modulated * radiate(beam.kx,theta_valid[:,0], beam.ky, theta_valid[:,1])
                to_receiver = torch.sum(modulated * radiate(beam.kx,theta_valid[:,0], beam.ky, theta_valid[:,1]), axis=1)
                received, beta = rayleigh_channel(to_receiver, sigma_c, sigma_n, lambda_txr)

                # present (perfect) Channel state estimation
                CSI = beta * torch.sum(direction * radiate(beam.kx,theta_valid[:,0], beam.ky, theta_valid[:,1]), axis=1) 
                # Receive and decode
                decoded=(dec(received, CSI))

                # radar target detection
                target_labels_ex = (target_labels.repeat(cpr,1).T).reshape(cpr*batch_size_per_epoch[epoch])
                target_ex = target.repeat(1,1,cpr).reshape(cpr*batch_size_per_epoch[epoch],max_target)

                received_rad,_ = radar_channel_swerling1(modulated,sigma_s, sigma_n, lambda_txr,rad_rec.k, phi_valid=phi_valid_ex, target=target_ex)
                if canc == 1:
                    received_radnn = (received_rad/encoded).to(device)
                else:
                    received_radnn = (received_rad).to(device)
                t_NN, angle = rad_rec(received_radnn, target_ex)
                
                if setbehaviour=="ESPRIT":
                    for i in range(int(batch_size_per_epoch[epoch])):
                        if target_labels[i]!=0:
                            angle[i,0,0:target_labels[i]] = esprit_angle_nns(received_rad[i*cpr:(i+1)*cpr,:],rad_rec.k,target_labels[i], cpr, 1)
                            angle[i,1,0:target_labels[i]] = esprit_angle_nns(received_rad[i*cpr:(i+1)*cpr,:],rad_rec.k[[1,0]],target_labels[i], cpr, 1)+np.pi/2
                angle[:,0,:] = target_ex * angle.clone()[:,0,:]
                angle[:,1,:] = target_ex * angle.clone()[:,1,:] 
                t_NN = torch.squeeze(t_NN) # -> BCEwithlogitsloss doesn't need sigmoid; works with logits
              
                if encoding=='counting':
                    t_NN = torch.mean(t_NN.reshape(batch_size_per_epoch[epoch],cpr,max_target),1) # Mean of LLRs
                if encoding=="onehot":
                    t_NN = torch.mean(t_NN.reshape(batch_size_per_epoch[epoch],cpr,max_target+1),1) # Mean of probabilities

                if setbehaviour=="sortall":
                    for l in range(int(batch_size_per_epoch[epoch]*cpr)):
                        t1 = torch.nonzero(torch.abs(angle[l,1,:])>0.1).to(device)
                        t0 = torch.nonzero(torch.abs(angle[l,1,:]) <= 0.1).to(device)
                        if t1.numel() > 0:
                            idx = torch.argsort(angle[l,0,torch.squeeze(t1,1)], descending=True).to(device)
                            if t0.numel() > 0:
                                idx = torch.squeeze(torch.cat((idx,torch.squeeze(t0,1))))
                            else:
                                idx = torch.squeeze(idx)
                        else:
                            idx = torch.squeeze(t0).to(device)
                        angle[l,0,:] = angle[l,0,idx]
                        angle[l,1,:] = angle[l,1,idx]

                if setbehaviour=="permute" or setbehaviour=="ESPRIT":
                    permuted_angle = permute( phi_valid_ex, angle, max_target,target_labels_ex)
                else:
                    permuted_angle = phi_valid_ex
                
                permuted_angle_shrunk = torch.mean(permuted_angle.reshape(batch_size_per_epoch[epoch],cpr,2,max_target),1)
                angle_shrunk = torch.mean(angle.reshape(batch_size_per_epoch[epoch],cpr,2,max_target),1)
                targ = torch.squeeze(torch.nonzero(permuted_angle_shrunk[:,0,0]))
                    
                if stage==1: # Training of angle estimation
                    loss = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                    angle_loss = 20*torch.mean(mse_loss(torch.squeeze(angle_shrunk), torch.squeeze(permuted_angle_shrunk)))
                    loss = (1-w_r)*loss.clone() + w_r*angle_loss #+ beamloss
                elif stage==3: # Training of target detection
                    loss = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                    if encoding=='onehot':
                        bloss = ce_loss(torch.squeeze(t_NN),torch.squeeze(target_labels))
                    else:
                        bloss = bce_loss(torch.squeeze(t_NN),torch.squeeze(target))
                        #bloss = bce_loss(torch.squeeze(t_NN),torch.squeeze(target))
                    loss = (1-w_r)*loss.clone() + w_r*(bloss)
                else:
                    loss = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                    angle_loss = 20*mse_loss(torch.squeeze(angle_shrunk), torch.squeeze(permuted_angle_shrunk))
                    #angle_loss = 20*torch.mean(mse_loss(torch.squeeze(angle_shrunk), torch.squeeze(permuted_angle_shrunk))*(target*torch.tensor([1,2,3], device=device) +1))
                    if encoding=='counting':
                        loss =  (1-w_r)*loss.clone() + (w_r)*torch.mean(bce_loss(torch.squeeze(t_NN),torch.squeeze(target)))# combine with radar detect loss
                    elif encoding=="onehot":
                        loss = (1-w_r)*loss.clone() + w_r*ce_loss(torch.squeeze(t_NN),torch.squeeze(target_labels))
                    loss = loss.clone() + (w_r)*angle_loss 
                        
                # compute gradients
                loss.backward() # seems to sometime throw complex warning for ESPRIT behaviour

                # run optimizer
                for elem in optimizer:
                    elem.step()
                
                # reset gradients
                for elem in optimizer:
                    elem.zero_grad()

        with torch.no_grad(): #no gradient required on validation data 
            # compute validation SER, BER, BMI
            direction = beam(np.pi/180*(torch.tensor([-20.0,20.0,30.0,50.0,0], device=device))).to(device)
            if plotting==True:
                cvalid = torch.zeros(N_valid*cpr)
            decoded_valid=torch.zeros((N_valid*cpr,int(torch.max(M))), dtype=torch.float32, device=device)
            y_valid_onehot = torch.eye(int(M), device=device)[y_valid]

            t_encoded = torch.matmul(torch.unsqueeze(enc(y_valid_onehot).to(device),1), torch.unsqueeze(direction,0))

            theta_valid = torch.zeros((N_valid,2)).to(device)
            theta_valid[:,0] = np.pi/180*(torch.rand(N_valid)*20+30)
            theta_valid[:,1] = np.pi/2
            theta_valid = theta_valid.repeat(1,1,cpr).reshape(cpr*N_valid,2)

            phi_valid = torch.zeros((N_valid,2,max_target)).to(device)
            phi_valid[:,0,:] = np.pi/180*(torch.rand((N_valid,max_target))*40-20) # paper: [-20 deg, 20 deg]
            phi_valid[:,1,:] = np.pi/2

            diffmin = 1*np.pi/180
            if max_target>1:
                for i in range(max_target-1):
                    dist = torch.abs(phi_valid[:,0,:i+1]-phi_valid[:,0,i+1].repeat(1,i+1).reshape(N_valid,i+1))
                    shift = torch.zeros_like(phi_valid[:,0])
                    a = torch.max(torch.zeros_like(dist)+(dist<diffmin),axis=1)[0]
                    shift[:,i+1] += 2*diffmin*torch.squeeze(torch.sign(phi_valid[:,0,i+1]))*torch.squeeze(a)
                    phi_valid[:,0,i+1] += shift[:,i+1]

            
            target_labels = torch.randint(max_target+1,(N_valid,)).to(device)#.type(torch.float32)

            target = torch.zeros((N_valid,max_target)).to(device)
            label_tensor = torch.zeros(max_target+1,max_target).to(device)
            for x in range(max_target+1):
                label_tensor[x] = torch.concat((torch.ones(x), torch.zeros(max_target-x)))
            target += label_tensor[target_labels]
            phi_valid[:,0,:] *= target
            phi_valid[:,1,:] *= target 

            target_onehot = torch.zeros((N_valid,max_target+1)).to(device)
            target_onehot[torch.arange(N_valid),target_labels] = 1
            
            if setbehaviour=="sortall" or setbehaviour=="sortinput":
                for l in range(int(N_valid)):
                    idx1 = torch.nonzero(phi_valid[l,0,:]).to(device)
                    idx0 = torch.nonzero(phi_valid[l,0,:] == 0).to(device)
                    if idx1.numel() > 0:
                        idx = torch.argsort(phi_valid[l,0,torch.squeeze(idx1,1)], descending=True).to(device)
                        if idx0.numel() > 0:
                            idx = torch.squeeze(torch.cat((idx,torch.squeeze(idx0,1))))
                        else:
                            idx = torch.squeeze(idx)
                    else:
                        idx = torch.squeeze(idx0).to(device)
                    phi_valid[l,0,:] = phi_valid[l,0,idx]
                    phi_valid[l,1,:] = phi_valid[l,1,idx]
    

            ## extent for cpr
            phi_valid_ex = phi_valid.repeat(1,cpr,1).reshape(cpr*N_valid,2,max_target)
            target_labels_ex = (target_labels.repeat(cpr,1).T).reshape(cpr*N_valid)

            target_ex = target.repeat(1,1,cpr).reshape(cpr*N_valid,max_target)

            encoded = torch.sum(t_encoded * radiate(beam.kx,theta_valid[:,0], beam.ky, theta_valid[:,1]), axis=1)
            channel, beta = rayleigh_channel(encoded, sigma_c, sigma_n, lambda_txr)

            received_rad,target_ex = radar_channel_swerling1(t_encoded,sigma_s, sigma_n, lambda_txr,rad_rec.k, phi_valid=phi_valid_ex, target=target_ex)
            received_radnn = received_rad.clone()
            t_NN, angle = rad_rec(received_radnn)
            angle[:,0,:] = target_ex * angle[:,0,:]  #set to 0 if no targets
            angle[:,1,:] = target_ex * angle[:,1,:]  #set to 0 if no targets

            if encoding == 'counting':
                t_NN = torch.mean(t_NN.reshape(N_valid,cpr,max_target),1)
                if t_NN.dim() > 1:
                    t_NN, _idx = torch.sort(t_NN,axis=1,descending=True)
            elif encoding == "onehot":
                t_NN = torch.mean(t_NN.reshape(N_valid,cpr,max_target+1),1)

            benchmark_angle_nn = torch.zeros(N_valid,2,max_target).to(device)
            if setbehaviour=="ESPRIT" or benchmark==1:
                for i in range(N_valid):
                    if target_labels[i]!=0:
                        c = received_rad[i*cpr:(i+1)*cpr,:]
                        benchmark_angle_nn[i,0,0:target_labels[i]] = esprit_angle_nns(c,rad_rec.k,target_labels[i], cpr,1)
                        benchmark_angle_nn[i,1,0:target_labels[i]] = esprit_angle_nns(c,rad_rec.k[[1,0]],target_labels[i], cpr,1)+np.pi/2

            # overall prob of detection
            if encoding== 'counting':
                targ = torch.nonzero(torch.squeeze(target)).to(device)
                targx = torch.nonzero(1-torch.squeeze(target))
                # detection rate
                prob_e_d = torch.sum(torch.round(torch.squeeze(t_NN))*(torch.squeeze(target)))/torch.sum(torch.squeeze(target)).to(device)
                # false alarm rate
                prob_f_d = torch.sum(torch.round(torch.squeeze(t_NN))*(1-torch.squeeze(target)))/torch.sum(1-torch.squeeze(target)).to(device)
            elif encoding== "onehot":
                t_NN_s = torch.zeros((N_valid, max_target+3)).to(device)
                t_NN_s[:,1:max_target+2] += t_NN
                num_targets = torch.sum(target_labels).to(device)
                m_targs = N_valid*max_target-num_targets
                #prob_e_d = torch.sum(torch.unsqueeze(target_labels,1)*t_NN)/num_targets
                t_help = torch.zeros_like(t_NN)
                t_help_f = torch.zeros_like(t_NN)
                t_help_argmax = torch.argmax(t_NN,axis=1)
                for l in range(N_valid):
                    if target_labels[l]>0:
                        t_help[l,1:target_labels[l]+1] = torch.arange(1,max_target+1)[0:target_labels[l]]
                    if target_labels[l]<max_target: 
                        t_help_f[l,target_labels[l]+1:] = torch.arange(1, max_target-target_labels[l]+1)
                #prob_e_d = 1/num_targets*torch.sum(t_help*torch.round(t_NN))
                #prob_f_d = 1/m_targs*torch.sum(torch.round(t_NN)*t_help_f)
                prob_e_d = 1/num_targets*torch.sum(t_help*torch.nn.functional.one_hot(t_help_argmax,max_target+1))
                prob_f_d = 1/m_targs*torch.sum(torch.nn.functional.one_hot(t_help_argmax,max_target+1)*t_help_f)
                
            if printing==True:
                print('Detect Probability of radar detection after epoch %d: %f' % (epoch, prob_e_d))            
                print('False Alarm rate of radar detection after epoch %d: %f' % (epoch, prob_f_d))
            logging.info('Detect Probability of radar detection after epoch %d: %f' % (epoch, prob_e_d))
            logging.info('False Alarm rate of radar detection after epoch %d: %f' % (epoch, prob_f_d))
            det_error[0].append(prob_e_d.detach().cpu().numpy())
            det_error[1].append(prob_f_d.detach().cpu().numpy())

            if setbehaviour=="sortall":
                for p in range(int(N_valid)):
                    t1 = torch.nonzero(torch.abs(angle[p,1,:])>0.1 ).to(device)
                    t0 = torch.nonzero(torch.abs(angle[p,1,:])<= 0.1).to(device) # sort all that have high elevation
                    if t1.numel() > 0:
                        idx = torch.argsort(angle[p,0,torch.squeeze(t1,1)], descending=True).to(device)
                        if t0.numel() > 0:
                            idx = torch.squeeze(torch.cat((idx,torch.squeeze(t0,1))))
                        else:
                            idx = torch.squeeze(idx)
                    else:
                        idx = torch.squeeze(t0).to(device)
                    angle[p,0,:] = angle[p,0,idx]
                    angle[p,1,:] = angle[p,1,idx]

            if (setbehaviour=="permute" or setbehaviour=="setloss") and max_target>1:
                permuted_angle = permute( phi_valid_ex, angle, max_target,target_labels_ex)
            else:
                permuted_angle = phi_valid_ex
            
            permuted_angle_shrunk = torch.mean(permuted_angle.reshape(N_valid,cpr,2,max_target),1)
            x_est = torch.squeeze(torch.nonzero(permuted_angle_shrunk[:,0,0])) # all timesteps that have at least one target
            if encoding=='counting':
                x_detect = torch.nonzero(t_NN*target > 0.5).to(device)[:,0] # targets that were present and were detected
            else:
                t_NN_x = torch.zeros((N_valid,max_target)).to(device)
                for n in range(max_target):
                    t_NN_x[:,n] = torch.sum(t_NN[:,n+1:],1)
                x_detect = torch.nonzero(t_NN_x*target > 0.5).to(device)[:,0]
            if max_target>1:
                benchmark_angle_nn = permute(benchmark_angle_nn, permuted_angle_shrunk, max_target,target_labels)
            rmse_benchmark.append(torch.sqrt(torch.mean(torch.abs((benchmark_angle_nn[x_detect,0,:] - permuted_angle_shrunk[x_detect,0,:]))**2)).detach().cpu().numpy())
            if setbehaviour == "ESPRIT":
                angle_shrunk = benchmark_angle_nn
            else:
                angle_shrunk = torch.mean(angle.reshape(N_valid,cpr,2,max_target),1)
            rmse_angle.append(torch.sqrt(torch.mean(torch.abs(torch.squeeze(angle_shrunk[x_detect,0,:])-torch.squeeze(permuted_angle_shrunk[x_detect,0,:]))**2)).detach().cpu().numpy())
     
            if printing==True:
                print('Angle estimation error after epoch %d: %f (rad)' % (epoch, rmse_angle[epoch]))
            logging.info('Angle estimation error after epoch %d: %f (deg) | %f (rad)' % (epoch, 180/np.pi*rmse_angle[epoch],rmse_angle[epoch]))

            # color map for plot
            if plotting==True:
                cvalid=y_valid

            CSI = beta * torch.sum(direction * radiate(beam.kx,theta_valid[:,0], beam.ky, theta_valid[:,1]), axis=1)  
            decoded_valid=dec(channel/CSI)
                
            validation_BER[epoch].append(BER((softmax(decoded_valid)), y_valid,M))
            validation_SERs[epoch] = SER((softmax(decoded_valid)), y_valid)

            if printing==True:
                print('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))  
                print('Validation SER after epoch %d: %f (loss %1.8f)' % (epoch, validation_SERs[epoch], loss.detach().cpu().numpy()))              
            
            logging.debug('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))
            logging.debug('Validation SER after epoch %d: %f (loss %1.8f)' % (epoch, validation_SERs[epoch], loss.detach().cpu().numpy()))

            BMI_exact[epoch]=BMI(M,(softmax(decoded_valid)), y_valid)
        
            if printing==True:
                print("BMI is: "+ str(torch.sum(BMI_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))
            logging.info("BMI is: "+ str(torch.sum(BMI_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))

            loss_ev = (-torch.sum(BMI_exact[epoch])-prob_e_d+prob_f_d).detach().cpu().numpy() + rmse_angle[epoch]
            if stage==1:
                loss_ev = rmse_angle[epoch]
            elif stage==3:
                loss_ev = prob_f_d - prob_e_d


            if epoch==0:
                enc_best=enc
                dec_best=dec
                best_epoch=0
                beam_best=beam
                rad_rec_best = rad_rec
                CSI_best = CSI
                loss_b = 10
            elif loss_ev<loss_b:
                enc_best=enc
                dec_best=dec
                best_epoch=epoch
                beam_best=beam
                rad_rec_best = rad_rec
                CSI_best =CSI

            validation_received.append((channel/CSI).detach().cpu().numpy())
            sent_all.append(t_encoded.detach().cpu().numpy())

            sig = torch.sum(direction * radiate(beam.kx,torch.tensor([0],device=device), beam.ky, torch.tensor([np.pi/2],device=device)), axis=1)**2 
            SNR = np.abs((sig*sigma_s**2/sigma_n**2).detach().cpu().numpy())
            CRB_azimuthi = 6*4/((2*np.pi)**2*np.cos(0*np.pi/180)**2*SNR*rad_rec.k[0].detach().cpu().numpy()*(rad_rec.k[0].detach().cpu().numpy()**2-1+1e-6))#/(rad_rec.k[1]*cpr)
            CRB_elevationi = 6*4/((2*np.pi)**2*np.cos(70*np.pi/180)**2*SNR*rad_rec.k[1].detach().cpu().numpy()*(rad_rec.k[1].detach().cpu().numpy()**2-1+1e-6))#/(rad_rec.k[0]*cpr)

            CRB_azimuth = np.minimum(CRB_azimuth,CRB_azimuthi)
            CRB_elevation = np.minimum(CRB_elevation,CRB_elevationi)

    constellations = cp.asarray(enc_best(torch.eye(int(M), device=device)).cpu().detach().numpy())

    logging.info("Constellation is: %s" % (str(constellations)))

    if plotting==True:
        decision_region_evolution = []
        mesh_prediction = (softmax(dec_best((torch.view_as_complex(torch.Tensor(meshgrid))).to(device)))) #*torch.exp(-1j*torch.angle(CSI_best[0]))
        decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
       
    print('Training finished')
    logging.info('Training finished')

    logging.info("SER obtained: %s" % (str(validation_SERs)))
    logging.info("BMI obtained: %s" % str(np.sum(BMI_exact.detach().cpu().numpy(),axis=1)))
      

    logging.info("CRB in azimuth is: "+str(CRB_azimuth))
    logging.info("CRB in elevation is: "+str(CRB_elevation))
    
    if plotting==True:
        if device=='cpu':
            plot_training(validation_SERs.cpu().detach().numpy(), np.array(validation_received[best_epoch]),cvalid,M, constellations, BMI, decision_region_evolution, meshgrid, BMI_exact.detach().cpu().numpy(), sent_all, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage,namespace=namespace, CRB=CRB_azimuth, antennas=rad_rec.k) 
        else:  
            plot_training(validation_SERs.cpu().detach().numpy(), validation_received[best_epoch],cvalid,M, constellations, BMI, decision_region_evolution, meshgrid, BMI_exact.detach().cpu().numpy(), sent_all, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage, namespace=namespace, CRB=CRB_azimuth, antennas=rad_rec.k) 

            path = figdir+"/plots"+".pkl"
            with open(path, 'wb') as fh:
                pickle.dump([validation_SERs.cpu().detach().numpy(), np.array(validation_received[best_epoch]),cvalid,M, constellations, BMI, decision_region_evolution, meshgrid, BMI_exact.detach().cpu().numpy(), sent_all, det_error, np.asarray(rmse_angle)], fh)
    if device=='cpu':
        return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,BMI_exact, det_error, cp.array(constellations))
    else:
        try:
            return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,BMI_exact, det_error, cp.asnumpy(constellations))
        except:
            return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,BMI_exact, det_error, (constellations))
