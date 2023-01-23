from Autoencoder.NN_classes import * #Encoder,Decoder,Radar_receiver,Radar_detect,Angle_est
from Autoencoder.functions import *
from Autoencoder.training_routine import train_network

logging.info("Running autoencoder_compare_cpr.py")

softmax = nn.Softmax(dim=1)


sigma_n=torch.tensor([0.1], dtype=float).to(device)
sigma_c=torch.tensor([1], dtype=float).to(device)
sigma_s = torch.tensor([1], dtype=float).to(device)

#path_list = ['set/none_new.pkl','set/sortphi_new.pkl','set/sortall_new.pkl','set/permute_new.pkl']
#set_methods = ['none','sortphi','sortall','permute','ESPRIT']
path_list = ['set/permute_new.pkl', 'set/permute_new_3.pkl']
set_methods = ['permute','permute_new','ESPRIT']
no_canc=[1,1,1,1,1,1]

num_nns = len(path_list)
m_angle_list =[]

logging.info(path_list)

N_valid = 1000
lambda_txr = 0.1
cpr_all = torch.arange(1,200,10).to(device)
#angles = np.pi/180*(torch.rand(N_angles)*40-20).to(device)
mse_plot = np.zeros((num_nns+1,len(cpr_all)))
CRB_azimuth = np.zeros(len(cpr_all))
p_detect = np.zeros((num_nns,len(cpr_all),2))

r_detector = CPU_Unpickler( open( path_list[0], "rb")).load()[3]
num_targets_trained = r_detector.targetnum
benchmark_angle = torch.zeros(N_valid,num_targets_trained).to(device)

target_labels = torch.randint(num_targets_trained+1,(N_valid,)).to(device) # torch.zeros(N_valid,dtype=torch.long).to(device)+3#
target_ij = torch.zeros((N_valid,num_targets_trained)).to(device)
label_tensor = torch.zeros(num_targets_trained+1,num_targets_trained).to(device)
for x in range(num_targets_trained+1):
    label_tensor[x] = torch.concat((torch.ones(x), torch.zeros(num_targets_trained-x)))
target_ij += label_tensor[target_labels]

angle_r = torch.zeros((N_valid,2), device=device)
angle_r[:,0] = np.pi/180*(torch.rand(N_valid)*20+30)
angle_r[:,1] = np.pi/180*(torch.rand(N_valid)*1+90) # between 90 and 100 deg


angle_t = torch.zeros((N_valid,2,num_targets_trained), device=device)#torch.deg2rad(torch.rand(N_valid)*40-20)
angle_t[:,0,:] += np.pi/180*(torch.rand((N_valid,num_targets_trained))*40-20).to(device)
angle_t[:,1,:] += np.pi/180*(torch.rand((N_valid,num_targets_trained))*10+80).to(device) # only one random thing since we add all values in ESPRIT

angle_t[:,0,:] *= target_ij
angle_t[:,1,:] *= target_ij

diffmin = 4*np.pi/180
if num_targets_trained>1:
    for i in range(num_targets_trained-1):
        t1 = angle_t[:,0,:i+1]
        t2 = angle_t[:,0,i+1].repeat(1,i+1).reshape(N_valid,i+1)
        dist = torch.abs(t1-t2)
        shift = torch.zeros_like(angle_t[:,0])
        a = torch.max(torch.zeros_like(dist)+(dist<diffmin),axis=1)[0]
        shift[:,i+1] += 2*diffmin*torch.squeeze(torch.sign(angle_t[:,0,i+1]))*torch.squeeze(a)
        angle_t[:,0,i+1] += shift[:,i+1]
    

for l in range(num_nns):
    if device=='cuda':
        enc, dec, beam, rad_rec = pickle.load( open( path_list[l], "rb" ) )
    else:
        enc, dec, beam, rad_rec =  CPU_Unpickler( open( path_list[l], "rb")).load()
    num_targets_trained = rad_rec.targetnum
    M = enc.M
    h=0
    f=0

    for cpr in cpr_all:
        targs=0
        tf = 0
        with torch.no_grad():
            t_nums = torch.zeros(N_valid,num_targets_trained).to(device)
            for N in range(N_valid):
                direction = beam(torch.deg2rad(torch.tensor([-20.0,20.0,30.0,50.0,0], device=device))).to(device)
                y_valid = torch.randint(int(M),(cpr,)).to(device)
                batch_labels_onehot = torch.eye(int(M), device=device)[y_valid]

                angle_receiver = angle_r[N].repeat(cpr,1).reshape(cpr,2).to(device)
                angle_target = angle_t[N].reshape(1,2,num_targets_trained).to(device)
                angle_target_ex = angle_t[N].repeat(cpr,1,1).reshape(cpr,2,num_targets_trained).to(device)

                decoded_valid=torch.zeros((1,int(torch.max(M))), dtype=torch.float32, device=device)

                y_valid_onehot = torch.eye(int(M), device=device)[y_valid]
                

                encoded_x = torch.unsqueeze(enc(y_valid_onehot).to(device),1)
                encoded = torch.matmul(encoded_x, torch.unsqueeze(direction,0)).to(device)

                to_receiver = torch.sum(encoded * radiate(beam.kx,angle_receiver[:,0], beam.ky, angle_receiver[:,1]), axis=1).to(device)
                channel, beta = rayleigh_channel(to_receiver, sigma_c, sigma_n, lambda_txr)
               
                target_i = target_ij[N,:].repeat(cpr).reshape(cpr*1,num_targets_trained)
                received_rad,target_i = radar_channel_swerling1(encoded, sigma_s, sigma_n, lambda_txr,rad_rec.k, phi_valid=angle_target_ex, target=target_i)
                
                if len(received_rad.size())==1:
                    received_rad = received_rad.reshape((1,torch.prod(rad_rec.k))).to(device)
                if l in no_canc:
                    t_NN, angle = rad_rec(received_rad)
                else:
                    t_NN, angle = rad_rec(received_rad/ encoded_x)
                angle = permute( angle, angle_target_ex, num_targets_trained, target_labels[N].repeat(cpr))
                angle_n = torch.mean(angle.reshape(1,cpr,2,num_targets_trained),axis=1)
                
                target_i = torch.squeeze(target_i).to(device)
                # overall prob of detection
                t_NN = t_NN.reshape(cpr,num_targets_trained)
                t_NN, _idx = torch.sort(t_NN,axis=1,descending=True)
                threshold = torch.tensor([0.5,0.5,0.5])
                t_NNx = torch.mean(torch.round(t_NN),0).to(device)
                decision = t_NNx > threshold
                t_nums[N] = decision
                targ_x = torch.mean(target_i.reshape(1,cpr,num_targets_trained),1).to(device)
                
                if target_labels[N] != 0 and t_NNx[0]>0.5:
                    benchmark_angle[N,0:target_labels[N]] = esprit_angle_nns(received_rad,rad_rec.k,target_labels[N],cpr)
                benchmark_angle[N] = permute( benchmark_angle[N,].reshape(1,num_targets_trained), angle_target[:,0,], num_targets_trained, target_labels[N].reshape(1))

                mse_angle_a = 0
                angle_n *= targ_x
                if target_labels[N] !=0 and t_NNx[0]>0.5:
                    mse_angle_a = (torch.abs((angle_n[:,0] - angle_target[:,0]))**2).detach().cpu().numpy()
                mse_angle_benchmark = (torch.abs((benchmark_angle[N,:target_labels[N]] - angle_target[:,0,:target_labels[N]]))**2).detach().cpu().numpy()
 
                CSI  = beta * torch.sum(direction * radiate(beam.kx,angle_receiver[:,0], beam.ky, angle_receiver[:,1]), axis=1).to(device)
                decoded_valid=dec(channel, CSI).to(device)
                    
                validation_BER=torch.mean(BER((softmax(decoded_valid)), y_valid,M)).item()
                validation_SERs= SER((softmax(decoded_valid)), y_valid).item()
                for number in range(target_labels[N]):
                    if target_labels[N]!=0 and t_NNx[number]>0.5:
                        targs += 1
                        mse_plot[l,h] += mse_angle_a.reshape(num_targets_trained)[number]
                        if l==num_nns-1:
                            mse_plot[l+1,h] += mse_angle_benchmark.reshape(target_labels[N])[number]
                            sig = torch.abs(torch.sum(direction * radiate(beam.kx,torch.tensor([0]).to(device), beam.ky, torch.tensor([np.pi/2],device=device)), axis=1)*(rad_rec.k[0].detach().cpu().numpy() * rad_rec.k[1].detach().cpu().numpy()))**2 
                            
            prob_e_d = torch.sum(torch.round(torch.squeeze(t_nums[:]))*(torch.squeeze(target_ij[:])))/torch.sum(torch.squeeze(target_ij[:,2])).to(device)
            # false alarm rate
            prob_f_d = torch.sum(torch.round(torch.squeeze(t_nums[:]))*(1-torch.squeeze(target_ij[:])))/torch.sum(1-torch.squeeze(target_ij[:,2])).to(device)
            p_detect[l,h,0] += prob_e_d.detach().cpu().numpy()
            p_detect[l,h,1] += prob_f_d.detach().cpu().numpy()
            if targs !=0:
                mse_plot[l,h] = mse_plot[l,h]/targs
                if l==num_nns-1:
                    mse_plot[l+1,h] = mse_plot[l+1,h]/targs
            h +=1
        f += 1
    logging.info("Simulation done: "+str(set_methods[l]))
print('Validation BER: '  + str(validation_BER) )
logging.info('Validation BER: %f'  % validation_BER)  
print('Validation SER: ' + str(validation_SERs) )
logging.info('Validation SER: %f' % validation_SERs)              

set_methods.append('ESPRIT')


BMI_exact = torch.sum(BMI(M,(softmax(decoded_valid)), y_valid)).item()

print("BMI is: "+str(BMI_exact)+" bit")
print("SER is: "+str(validation_SERs))

logging.info("BMI is: %f bit" % BMI_exact)


print("Detection probability is: "+str(prob_e_d))
print("Angle mse is: "+ str(mse_angle_a))


plt.figure()
for l in range(num_nns):
    plt.plot(cpr_all.detach().cpu().numpy(),p_detect[l,:,0], marker='x', label=r"$P_{detect}$")
    plt.plot(cpr_all.detach().cpu().numpy(),p_detect[l,:,1], marker='o', label=r"$P_{error}$")
plt.xlabel("upsampling factor")
plt.ylabel("Probability")
plt.legend(loc=3)
plt.ylim(0,1)
plt.grid()
plt.savefig(figdir+"/Pef"+namespace+".pdf")
label = ["cpr"]
for i in range(num_nns):
    label.append("Pd"+set_methods[i])
for i in range(num_nns):
    label.append("Pf"+set_methods[i])
t= np.zeros((num_nns*2+1,len(cpr_all.detach().cpu().numpy())))
t[0] = cpr_all.detach().cpu().numpy()
t[1:num_nns+1] = p_detect[:,:,0]
t[num_nns+1:] = p_detect[:,:,1]
save_to_txt(t,"pef_vs_cpr",label)


plt.figure()
for l in range(num_nns+1):
    plt.plot(cpr_all.detach().cpu().numpy(),np.sqrt(mse_plot[l]), marker='x', label=set_methods[l])
plt.xlabel("upsampling factor")
plt.ylabel("RMSE (rad)")
plt.legend(loc=3)
plt.yscale('log')
plt.grid()
plt.savefig(figdir+"/mse_compare"+namespace+".pdf")

logging.info("Simulation finished!")

label=["cpr"]
for i in range(num_nns):
    label.append(set_methods[i])
label.append("ESPRIT")
t= np.zeros((num_nns+3,len(cpr_all.detach().cpu().numpy())))
t[0] = cpr_all.detach().cpu().numpy()
t[1:num_nns+2] = np.sqrt(mse_plot)
save_to_txt(t,"rmse_vs_cpr",label)
