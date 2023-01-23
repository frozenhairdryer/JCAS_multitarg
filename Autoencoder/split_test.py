import datetime
import pickle
import sys

""" ## choice controls the kind of simulation ##
choice = 1    : Train for 3 targets
choice = 2    : Train for 1 target
"""
# Switching of the encoding has to be performed in training_routine_multitarget.py

begin_time = datetime.datetime.now()
#### Enable setting arguments from command line
if len(sys.argv)==1:
    choice = 3
elif len(sys.argv)==2:
    choice = int(sys.argv[1])
elif len(sys.argv)==7:
    choice=0
    from training_routine import *
    M = torch.tensor([int(sys.argv[1])], dtype=int).to(device)
    sigma_n=torch.tensor([float(sys.argv[2])], dtype=float, device=device)
    sigma_c=torch.tensor([float(sys.argv[3])]).to(device)
    sigma_s=torch.tensor([float(sys.argv[4])]).to(device)
    max_target = int(sys.argv[5])
    setbehaviour = sys.argv[6]
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling 1 parameter = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=np.array([50,20,0.001,1]),w_r=0.9,max_target=max_target,stage=1, plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=np.array([50,20,0.001,1]),w_r=0.9,max_target=max_target,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=np.array([50,20,0.001,1]),w_r=0.9,max_target=max_target,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour=setbehaviour, namespace=namespace) 
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==8:
    choice=0
    from training_routine import *
    M = torch.tensor([int(sys.argv[1])], dtype=int).to(device)
    sigma_n=torch.tensor([float(sys.argv[2])], dtype=float, device=device)
    sigma_c=torch.tensor([float(sys.argv[3])]).to(device)
    sigma_s=torch.tensor([float(sys.argv[4])]).to(device)
    max_target = int(sys.argv[5])
    setbehaviour = sys.argv[6]
    wr = float(sys.argv[7])
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling 1 parameter = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=np.array([50,20,0.001,1]),w_r=wr,max_target=max_target,stage=1, plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=np.array([50,20,0.001,1]),w_r=wr,max_target=max_target,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=np.array([50,20,0.001,1]),w_r=wr,max_target=max_target,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour=setbehaviour, namespace=namespace) 
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
else:
    pass


if choice == 0:
    pass
elif choice == 1:
    from training_routine import *
    logging.info("One simulation with 1 Target")
    M=torch.tensor([4], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([sigma_n]).to(device)
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=1,stage=1, plotting=True,setbehaviour="none", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=1,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 2:
    from training_routine import *
    logging.info("One simulation with 1 Target")
    M=torch.tensor([4], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([sigma_n]).to(device)
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 3:
    from training_routine import *
    logging.info("One simulation with 3 Targets and permute set behavior")
    M=torch.tensor([8], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([10*sigma_n]).to(device)
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=3,stage=1, plotting=True,setbehaviour="permute", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=3,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="permute", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,20,0.001,1]),w_r=0.9,max_target=3,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="permute", namespace=namespace)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
logging.info("Training duration is" + str(datetime.datetime.now()-begin_time))