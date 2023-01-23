from imports import *

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # fix renaming issues
        if name[:14]=='Radar_receiver':
            name= 'Radar_receiver'
        elif name[:9]=='Angle_est':
            name="Angle_est"
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

def SER(predictions, labels):
    """Calculates Hard decision SER
    
    Args:
    predictions (float): NN autoencoder output; prediction one-hot vector for symbols
    labels (int): actually sent symbols (validation symbols)   

    Returns:
        SER (float) : Symbol error rate

    """
    s2 = torch.argmax(predictions, 1).to(device)
    return (torch.sum( s2!= labels))/predictions.shape[0] # Limit minimum SER to 1/N_valid with +1

def BER(predictions, labels,m):
    """Calculates Hard decision bit error rate
    
    Args:
    predictions (float): NN autoencoder output; prediction one-hot vector for symbols
    m (int): number of modulation symbols per user
    labels (int): actually sent symbols (validation symbols)   

    Returns:
        ber (float) : bit error rate

    """
    # Bit representation of symbols
    binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,m.detach().cpu().numpy(),dtype='uint8')), (-1,8)), device=device)
    binaries = binaries[:,int(8-torch.log2(m)):]
    y_valid_binary = binaries[labels]
    pred_binary = binaries[torch.argmax(predictions, axis=1),:]
    ber=torch.zeros(int(torch.log2(m)), device=device, requires_grad=True)
    ber = 1-torch.mean(torch.isclose(pred_binary, y_valid_binary,rtol=0.5)+0.0,axis=0, dtype=float)
    return ber

def BMI(M, my_outputs, mylabels):
    """Calculation of Generalized mutual information
    
    Args:
    M ( int): number of modulation symbols per user
    my_outputs (float): Symbol probabilities
    mylabels (int): validation labels of sent symbols    

    Returns:
        r_signal: signal after channel, still upsampled

    """
    # bmi calculation
    gmi=torch.zeros(int(torch.log2(M)), device=device)
    binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,M.detach().cpu().numpy(),dtype='uint8')), (-1,8)),dtype=torch.float32, device=device)
    binaries = binaries[:,int(8-torch.log2(M).detach().cpu().numpy()):]
    b_labels = binaries[mylabels].int()
    # calculate bitwise estimates
    bitnum = int(torch.log2(M))
    for bit in range(bitnum):
        pos_0 = torch.where(binaries[:,bit]==0)[0]
        pos_1 = torch.where(binaries[:,bit]==1)[0]
        est_0 = torch.sum(torch.index_select(my_outputs,1,pos_0), axis=1) 
        est_1 = torch.sum(torch.index_select(my_outputs,1,pos_1), axis=1)  # increase stability
        llr = torch.log((est_0+1e-12)/(est_1+1e-12)+1e-12)
        gmi[bit]=1-1/(len(mylabels))*torch.sum(torch.log2(torch.exp((2*b_labels[:,bit]-1)*llr)+1+1e-12), axis=0)
    return gmi.flatten()

def rayleigh_channel(sigIN, sigma_c, sigma_n, lambda_txr):
    """ Apply Rayleigh Channel Model
        parameters:
        sigIN : complex input signal into channel
        sigma_c : rayleigh fading; beta~CN(0,sigma_c^2)
        sigma_n : AWGN standard deviation
        lambda_txr : wavelength of carrier signal 
        theta_valid : Angle at which the receiver is present

        output:
        sigOUT : output signal
        beta : current fading parameter
         
    """
    beta = ((torch.randn(len(sigIN))+1j*torch.randn(len(sigIN))).to(device)*sigma_c/np.sqrt(2)).type(torch.complex64).to(device) # draw one beta value per transmission
    sigOUT = sigIN.permute(*torch.arange(sigIN.ndim - 1, -1, -1)) * beta
    noise = sigma_n/np.sqrt(2)*(torch.randn(sigOUT.size())+1j*torch.randn(sigOUT.size())).to(device) #add noise
    sigOUT += noise
    return torch.squeeze(sigOUT), beta 

def radar_channel_swerling1(sigIN, sigma_s, sigma_n, lambda_txr,k_antenna, phi_valid=0, target=torch.tensor([1])):
    """ Apply Rice Channel Model for the radar detection
        parameters:
        sigIN : complex input signal into channel (True antenna output without steering vectors)
        sigma_s : variance of radar cross section of target; alph~CN(0,sigma_r^2); swerling 1 model
        sigma_n : AWGN standard deviation
        lambda_txr : wavelength of carrier signal 
        theta_valid : Angle at which the radar target might be present
        k_antenna : number of antennas for radar receiver
        target : We can specify at which timesteps we want targets to appear

        output:
        sigOUT : output signal
        target (bool) : is a target present
         
    """
    d=lambda_txr/2 # distance between antennas is exaclty lambda/2
    k_dim = torch.prod(k_antenna).to(device)
    sent = (torch.zeros(sigIN.size()[0],k_dim)+0j).to(device)
    s=torch.zeros_like(sent)+0j
    max_targ = phi_valid.size()[2]
    alpha = ((torch.randn(sigIN.size()[0],max_targ).to(device)+1j*torch.randn(sigIN.size()[0],max_targ).to(device))*sigma_s/np.sqrt(2)*target).type(torch.complex64).to(device)
    for targ in range(max_targ):
        aTX_RX = torch.unsqueeze(radiate(k_antenna[0].to(device),phi_valid[:,0,targ].to(device),k_antenna[1].to(device), phi_valid[:,1,targ]),2).to(device)
        sent = sent.clone() + torch.squeeze(torch.matmul(torch.matmul(aTX_RX,torch.transpose(aTX_RX,1,2)),torch.unsqueeze(sigIN,2)))*torch.unsqueeze(alpha[:,targ],1)
    sigOUT = sigma_n/np.sqrt(2)*(torch.randn((sigIN.size()[0],k_dim))+1j*torch.randn((sigIN.size()[0],k_dim))).to(device) #add noise
    SNR = 10*torch.log10(torch.mean(torch.abs(alpha**2))/torch.mean(torch.sum(torch.abs(sigOUT)**2,axis=1))).to(device)
    sigOUT += sent
    return torch.squeeze(sigOUT), target

def radiate(kx,phix,ky=1,phiy=torch.tensor([np.pi/2], device=device)):
    ## Radiate from a kx*ky antenna array that is oriented in y-z-direction (phix is elevation, phiy is azimuth)
    kx_a=torch.arange(kx, device=device)+1
    ky_a=torch.arange(ky, device=device)+1
    radiatedx = torch.exp(1j*np.pi*torch.kron(torch.unsqueeze(kx_a,1),torch.unsqueeze(torch.sin(phix)*torch.sin(phiy),0)))#torch.unsqueeze(kx,0)*np.sin(torch.unsqueeze(phi,1))).T
    radiatedy = torch.exp(1j*np.pi*torch.kron(torch.unsqueeze(torch.cos(phiy),1),ky_a))
    radiated = (torch.unsqueeze(radiatedx.T,2) * torch.unsqueeze(radiatedy,1))
    radiated = torch.reshape(radiated, (radiated.size()[0],-1))
    return radiated


def permute(x_i,y_i, max_target, targets):
    """Permute the NN output vector x_i, so that its quared error is minimal
    Args:
    x_i (Nx2xmax_target) (float): estimated angles (time domain)
    y_i (Nx2x,max_target) (float): target angles
    max_target (int): Maximum of targets that are to be detected
    targets (N) (int): Holds number of targets present for each time step

    Returns:
        permuted (like x_i): permuted angles so that the vector matches y_i best 

    """
    if (x_i.size()==y_i.size()) and len(x_i.size())==2:
        x_i = torch.unsqueeze(x_i,1).to(device)
        y_i = torch.unsqueeze(y_i,1).to(device)
    p_targets = np.arange(max_target)
    permuted = torch.zeros_like(torch.as_tensor(x_i)).to(device)
    perm = list(permutations(p_targets, r=max_target))
    for n in range(len(targets)):
        if targets[n]!=0:
            c = (torch.permute(x_i[n,:,perm],(1,0,2))-torch.squeeze(y_i[n,:]))**2
            c1 = torch.reshape(c, (len(perm),-1))
            b = torch.sum(c1,axis=1)
            ind = torch.argmin(b, axis=0)
            permuted[n] = x_i[n,:,perm[ind]]
    return permuted

def esprit_angle_nns(x_in,n,L=torch.tensor([1],device=device),cpr=1):
    """
    Esprit algorithm to estaimate angles from cpr samples of all N antennas
    x_in (cpr,N): input samples
    n (kx,ky): number antennas
    L : number targets, optional
    cpr : upsampling factor
    OUT:
    angles: estimated angles in rad
    """
    i_ss = int(cpr*n[1])
    if i_ss>1:
        x_i = torch.reshape(x_in, (cpr, n[0], n[1]))
        x_i = torch.transpose(x_i, 1,2)
        x_i = torch.transpose(x_i, 1,0)
        x_i = torch.transpose(torch.reshape(x_i,(cpr*n[1],n[0])),0,1)
    N = n[0]
    angles = torch.zeros(torch.max(L))
    if i_ss == 1:
        x_i = torch.reshape(x_in,(n[0],i_ss))
        N=n[0]-L
        f = [x_i[idx:n[0]+idx-L] for idx in range(L+1)]
        x_ij = torch.squeeze(torch.stack(f,axis=1)) # single snapshot ESPRIT: create Hankel Matrix to prevent ambiguity
    else:
        x_ij = x_i
    R = x_ij @ torch.transpose(torch.conj(x_ij),0,1)
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = torch.linalg.eig(R)
    S = U[:,0:L]
    Phi = torch.linalg.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs, _ = torch.linalg.eig(Phi)
    DoAsESPRIT = torch.arcsin(torch.angle(eigs)/np.pi)
    angles = DoAsESPRIT
    return angles

# Saving to output file
def save_to_txt(x_in,s="name", label='signal'):
    """
    Save arrays to txt files to build easy plots in pgfplots
    x_in (sig_length, sig_numbers): values to save
    s : filename
    label : column labels
    """
    result = [
                l+" " for l in label
    ]
    result.append("\n")
    x_in = np.transpose(x_in)
    for l in x_in:
        for t in l:
            result.append(str(t)+" "
                        )
        result.append("\n")
    with open(figdir+'/'+str(s)+'.txt'.format(), 'w') as f:
        f.writelines(result)


def plot_training(SERs,valid_r,cvalid,M, const, GMIs_appr, decision_region_evolution, meshgrid, gmi_exact, encoded, detect_error, d_angle,benchmark = None, stage=None, namespace="", CRB=0, antennas=torch.tensor([16,1], device=device)):
    """Creates mutliple plots in /figures for the best implementation
    
    Args:
    SERs (float): Hard-decision Symbol error rates
    valid_r (float, float) : Received signal (decoder input) 
    cvalid (int) : integer specifying symbol number (range is 0...M_all) -> colorcoded symbols 
    M (int): number of modulation symbols per user
    const (complex), len=M_all : resulting constellation (channel input)
    GMIs_appr (float) : GMI estimate from SERs
    decision_region_evolution (int) : grid containing ints denoting the corresponding symbol
    meshgrid (float): grid on which decision_region_evolution is based
    constellation_base (complex) [len(M)]: contains all possible outputs for all encoders
    gmi_exact (float): GMI calculated from LLRs -> exact value 
       
    Plots:
     * SERs vs. epoch
     * GMIs (gmi_exact) vs. epoch
     * scatterplot const
     * scatterplot valid_r as complex number
     * scatterplot complex decision regions together with received signal
     * scatterplots for base constellations

    Returns:
        none

    """
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 10,
    'figure.max_open_warning' : 100
    })
    
    #plt.set_visible(False)
    cmap = matplotlib.cm.tab20
    base = plt.cm.get_cmap(cmap)
    color_list = base.colors
    new_color_list = np.array([[t/2 + 0.49 for t in color_list[k]] for k in range(len(color_list))])

    min_SER_iter = np.argmin(SERs)
    max_GMI = len(SERs)-1 #last iter   #np.argmax(np.sum(gmi_exact, axis=1))
    ext_max_plot = 1.2#*np.max(np.abs(valid_r[int(min_SER_iter)]))

    print('Minimum mean SER obtained: %1.5f (epoch %d out of %d)' % (SERs[min_SER_iter], min_SER_iter, len(SERs)))
    print('Maximum obtained BMI: %1.5f (epoch %d out of %d)' % (np.sum(gmi_exact[max_GMI]),max_GMI,len(SERs)))
    print('The corresponding constellation symbols are:\n', const)

    plt.figure("SERs "+str(stage),figsize=(3.5,3.5))
    plt.plot(SERs,marker='.',linestyle='--',markersize=2)
    plt.plot(min_SER_iter,SERs[min_SER_iter],marker='o',markersize=3,c='red')
    plt.annotate('Min', (0.95*min_SER_iter,1.4*SERs[min_SER_iter]),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('SER')
    plt.grid(visible=True,which='both')
    plt.title('SER on Validation Dataset')
    plt.savefig(figdir+"/Sers"+str(stage)+namespace+".pdf")

    plt.figure("GMIs "+str(stage),figsize=(3,2.5))
    for num in range(len(gmi_exact[0,:])):
        if num==0:
            t=gmi_exact[:,num]
            plt.fill_between(np.arange(len(t)),t, alpha=0.4)
        else:
            plt.fill_between(np.arange(len(t)),t,(t+gmi_exact[:,num]),alpha=0.4)
            t+=gmi_exact[:,num]
    plt.plot(t, label='GMI')
    plt.plot(argmax(t),max(t),marker='o',c='red')
    plt.annotate('Max', (0.95*argmax(t),0.9*max(t)),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('GMI')
    plt.ylim(0,np.round(np.max(t))+1)
    plt.xlim(0,len(t)-1)
    plt.grid(visible=True,which='both')
    plt.title('GMI on Validation Dataset')
    plt.tight_layout()
    plt.savefig(figdir+"/gmis"+str(stage)+namespace+".pdf")

    try:
        constellations = const.get().flatten()
    except:
        constellations = np.asarray(const).flatten()
    bitmapping=[]
    torch.prod(M)
    int(torch.prod(M))
    helper= np.arange((int(torch.prod(M))))
    for h in helper:
        if M==16:
            bitmapping.append(format(h, ('04b')))
        elif M==8:
            bitmapping.append(format(h, ('03b')))
        else:
            t = int(np.log2(M))
            str_b = '0' + str(t) + 'b'
            bitmapping.append(format(h, (str_b)))

    plt.figure("constellation "+str(stage), figsize=(5,5))
    plt.scatter(np.real(constellations),np.imag(constellations),c=range(np.product(M.cpu().detach().numpy())), cmap='tab20',s=50)
    for i in range(len(constellations)):
        plt.annotate(bitmapping[i], (np.real(constellations)[i], np.imag(constellations)[i]))
    
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-1.5,1.5))
    plt.ylim((-1.5,1.5))
    plt.grid(visible=True,which='both')
    #plt.title('Constellation')
    plt.tight_layout()
    plt.savefig(figdir+"/constellation"+str(stage)+namespace+".pdf")
    t = np.zeros((3,len(constellations)))
    t[0] = np.real(constellations)
    t[1] = np.imag(constellations)
    t[2] = bitmapping
    save_to_txt(t,s="constellation",label=["real","imag","bitmapping"])


    val_cmplx=np.asarray(valid_r)
    plt.figure("Received signal"+str(stage),figsize=(2.7,2.7))
    plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000].cpu().detach().numpy(), cmap='tab20',s=2)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid(visible=True)
    plt.title('Received')
    plt.tight_layout()
    plt.savefig(figdir+"/received"+str(stage)+namespace+".pdf")
    
    plt.figure("Decision regions"+str(stage), figsize=(5,3))
    for num in range(len(M)):
        plt.subplot(1,len(M),num+1)
        decision_scatter = np.argmax(decision_region_evolution[num], axis=1)
        grid=np.asarray(meshgrid)
        if num==0:
            plt.scatter(grid[:,0], grid[:,1], c=decision_scatter,s=2,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:int(M[num])]))
        else:
            plt.scatter(grid[:,0], grid[:,1], c=decision_scatter,s=2,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[int(M[num-1]):int(M[num-1])+int(M[num])]))
        plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000].cpu().detach().numpy(), cmap='tab20',s=2)
        plt.axis('scaled')
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.title('Decoder %d' % (num+1))
    plt.tight_layout()
    plt.savefig(figdir+"/decision_regions"+str(stage)+namespace+".pdf")

    phi = np.arange(-np.pi,np.pi,np.pi/180)
    theta = np.arange(0,np.pi,np.pi/90)

    a_phi = radiate(antennas[0],torch.tensor(phi, device=device),antennas[1]).detach().cpu().numpy().T
    a_theta = radiate(antennas[0],torch.tensor([0], device=device),antennas[1],torch.tensor(theta, device=device)).detach().cpu().numpy().T

    plt.figure("Beampattern Azimuth", figsize=(5,3))
    enc1 = np.array(encoded)[max_GMI]
    E_phi = 10*np.log10(np.abs(enc1[0] @ a_phi )**2+1e-9)
    
    plt.plot(phi*180/np.pi, E_phi,label='epoch '+str(max_GMI)+" stage "+str(stage))
    plt.fill_between(phi[160:201]*180/np.pi,E_phi[160:201],-50,alpha=0.2,label='radar target')
    plt.fill_between(phi[210:231]*180/np.pi,E_phi[210:231],-50,alpha=0.2,label='receiver')
    plt.xlabel("Angle (deg) azimuth")
    plt.ylabel("radiated Power (dB)")
    plt.xlim(-90,90)
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(figdir+"/E_phi_a"+namespace+".pdf")


    ### Animation Beamforming
    #from matplotlib.animation import FuncAnimation
    #plt.style.use('seaborn-pastel')
    """ filenames = []
    for d in range(len(encoded)):
        enc1 = np.array(encoded)[d]
        E_phi = 10*np.log10(np.mean(np.abs(enc1 @ a_phi )**2, axis=0))
        # plot the line chart
        plt.figure(figsize=(6,4))
        plt.plot(phi*180/np.pi, E_phi,label='epoch no. '+str(d))
        plt.fill_between(phi[160:201]*180/np.pi,E_phi[160:201],-50,alpha=0.2,label='radar target')
        plt.fill_between(phi[210:231]*180/np.pi,E_phi[210:231],-50,alpha=0.2,label='receiver')
        plt.xlabel("Angle (deg)")
        plt.ylabel("radiated Power (dB)")
        plt.xlim(-90,90)
        plt.ylim(-40,5)
        plt.legend(loc=3)
        plt.grid(visible=True)
        plt.tight_layout()
        
        # create file name and append it to a list
        filename = figdir+f'/{d}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()# build gif

    with imageio.get_writer(figdir+"/beamform"+str(stage)+namespace+".gif", mode='I', fps=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename) """

    try:
        plt.figure("detection error")
        plt.plot(detect_error[0],marker='.',linestyle='--',markersize=2, label="detection prob "+str(stage))
        plt.plot(np.argmax(detect_error[0]),np.max(detect_error[0]),marker='o',markersize=3,c='red')
        plt.annotate('Max', (np.argmax(detect_error[0]),np.max(detect_error[0])),c='red')

        plt.plot(detect_error[1],marker='.',linestyle='--',markersize=2, label="False alarm rate "+str(stage))
        plt.plot(np.argmin(detect_error[1]),np.min(detect_error[1]),marker='o',markersize=3,c='red')
        plt.xlabel('epoch no.')
        plt.ylabel(r'$P$')
        plt.grid(visible=True,which='both')
        plt.legend(loc=2)
        plt.title('Detection probability on Validation Dataset')
        plt.tight_layout()
        plt.savefig(figdir+"/PeDetect"+namespace+".pdf")
    except:
        pass


    f = np.zeros((3,len(d_angle)))
    f[0] = np.arange(len(d_angle))
    f[1] = detect_error[0]
    f[2] = detect_error[1]
    save_to_txt(f,s="Pd-Pf"+str(stage),label=['epoch',"Pd","Pf"])

    
    try:
        plt.figure("Angle estimate",figsize=(6,3.5))
        x=np.arange(len(d_angle))
        plt.plot(x,np.abs(d_angle), label="cycle"+str(stage))
        try:
            plt.plot(x,np.abs(benchmark), label="ESPRIT")
        except:
            pass
        plt.plot(x,np.repeat(np.sqrt(CRB),len(d_angle)),'--',label="CRB")
        plt.xlabel('epoch no.')
        plt.ylabel(r"RMSE (rad)")
        plt.grid(visible=True)
        #plt.ylim(0,0.13)
        plt.xlim(0,len(d_angle)-1)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(figdir+"/Angle_est"+namespace+".pdf")
        saved = np.zeros((3,len(d_angle)))
        saved[0] = x
        saved[1] = np.abs(d_angle)
        saved[2] = np.repeat(np.sqrt(CRB),len(d_angle))
        save_to_txt(saved,s="RMSEangle",label=["epoch","RMSE","CRB"])
    except:
        pass


    

