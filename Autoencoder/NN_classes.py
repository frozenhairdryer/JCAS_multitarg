from logging import raiseExceptions
from imports import *


class Encoder(nn.Module):
    def __init__(self,M):
        super(Encoder, self).__init__()
        self.M = torch.as_tensor(M, device=device)
        self.K = 16
        # Define Transmitter Layer: Linear function, M icput neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(self.M,2*self.M, device=device) 
        self.fcT2 = nn.Linear(2*self.M, 2*self.M,device=device)
        self.fcT3 = nn.Linear(2*self.M, 2*self.M,device=device) 
        self.fcT5 = nn.Linear(2*self.M, 2,device=device)
        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()  # in paper: LeakyReLU for hidden layers    

    def forward(self, x):
        # compute output
        out = self.activation_function(self.fcT1(x))
        out = self.activation_function(self.fcT2(out))
        out = self.activation_function(self.fcT3(out))
        encoded = self.activation_function(self.fcT5(out))
        # compute normalization factor and normalize channel output
        norm_factor = torch.sqrt(torch.mean(torch.abs((torch.view_as_complex(encoded)).flatten())**2)) # normalize mean squared amplitude to 1
        modulated = torch.view_as_complex(encoded)/norm_factor
        return modulated
    
def PSK_encoder(idx,M):
    if max(idx)>=M or min(idx)<0:
        raiseExceptions("Modulation format not high enough")
    symbol_idx = torch.arange(M).to(device)
    symbols = torch.exp(1j*2*np.pi*symbol_idx/M)
    encoded = symbols[idx]
    return encoded


class Decoder(nn.Module):
    def __init__(self,M):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 icput neurons (real and imaginary part), M output neurons (symbols)
        self.M = torch.as_tensor(M, device=device)
        self.K= 16
        self.fcR1 = nn.Linear(2,2*self.M,device=device) 
        self.fcR2 = nn.Linear(2*self.M,2*self.M,device=device) 
        self.fcR3 = nn.Linear(2*self.M,2*self.M,device=device)
        self.fcR5 = nn.Linear(2*self.M, self.M,device=device) 
        self.activation_function = nn.ELU()
    

    def forward(self, x, CSI=1):
        # compute output
        x_prep = x/CSI # divide by current channel state information = complex gain of signal; this might lead to noise amplification
        x_real = torch.view_as_real(x_prep).float()
        out = self.activation_function(self.fcR1(x_real))
        out = self.activation_function(self.fcR2(out))
        out = self.activation_function(self.fcR3(out))
        logits = self.activation_function(self.fcR5(out))
        lmax =  torch.max(logits, 1).values
        #for l in range(len(logits)):
        logits = torch.transpose(torch.transpose(logits,0,1) - lmax,0,1) # prevent very high and only low values -> better stability
        return logits+1

class Beamformer(nn.Module):
    """ Transforms the single transmit signal into a tensor of transmit signals for each Antenna
    beamforming is necessary if multiple receivers are present or additional Radar detection is implemented.
    Learning of an appropriate beamforming:
    
    input paramters:
        [theta_min, theta_max] : Angle Interval of Radar target
        [phi_min, phi_max] : Angle interval of receiver
        theta_last : last detected angle of target

    output parameters:
        out : phase shift for transmit antennas
     """
    def __init__(self,kx,ky=1):
        super(Beamformer, self).__init__()
        self.kx =torch.as_tensor(kx) # Number of transmit antennas in x-direction;
        self.ky =torch.as_tensor(ky) # Number of transmit antennas in y-direction;
    
        self.fcB1 = nn.Linear(5,self.kx, device=device)
        self.fcB2 = nn.Linear(self.kx,self.kx, device=device)
        self.fcB3 = nn.Linear(self.kx,self.kx*2, device=device)
        self.fcB4 = nn.Linear(self.kx*2,self.kx*2, device=device) # linear output layer

        if ky>1:
            self.fcA1 = nn.Linear(5,self.ky, device=device)
            self.fcA2 = nn.Linear(self.ky,self.ky, device=device)
            self.fcA3 = nn.Linear(self.ky,self.ky*2, device=device)
            self.fcA4 = nn.Linear(self.ky*2,self.ky*2, device=device) # linear output layer
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def forward(self, Theta):
        out = self.activation_function(self.fcB1(Theta)).to(device)
        out = self.activation_function(self.fcB2(out)).to(device)
        out_2 = self.activation_function(self.fcB3(out)).to(device)
        outx = torch.view_as_complex(torch.reshape(self.activation_function(self.fcB4(out_2)),(self.kx,1,2))).to(device)
        
        outy = torch.tensor([[1+0j]]).to(device)
        if self.ky>1:
            out = self.activation_function(self.fcA1(Theta)).to(device)
            out = self.activation_function(self.fcA2(out)).to(device)
            out_2 = self.activation_function(self.fcA3(out)).to(device)
            outy = torch.view_as_complex(torch.reshape(self.activation_function(self.fcA4(out_2)),(1,self.ky,2))).to(device)
            
        out_allp = outx @ outy  
        d = torch.sum(torch.abs(out_allp)**2).to(device) # Power norm so that we can compare Antenna configurations
        out_all = torch.reshape(out_allp,(-1,))/torch.sqrt(d)
        return out_all

class Radar_receiver(nn.Module):
    """ Detects radar targets and estimates positions (angles) at which the targets are present.
    
    input paramters:
        k: number of antennas of radar receiver (linear array kx1)

    output parameters:
        detect: bool whether radar target is present
        angle: estimated angle(s)
        uncertain: uncertainty of angle estimate

     """
    def __init__(self,kx,ky,max_target=1, encoding="counting"):
        super(Radar_receiver, self).__init__()
        self.k =torch.as_tensor([kx,ky]) # Number of transmit antennas; Paper k=16
        self.detect_offset = 0
        self.targetnum = max_target
        self.encoding = encoding
        self.rad_detect = Radar_detect(k=self.k,max_target=max_target, encoding=encoding).to(device)
        self.rad_angle_est = Angle_est(k=self.k,max_target=max_target).to(device)

    
    def forward(self, c_k, targets=None):
        detect = self.rad_detect(c_k)
        Pf =0.01 # maximum error probability
        if self.encoding=="counting" or self.encoding=="sum of onehot":
            if targets!=None:
                x = torch.nonzero((1-targets))
                t = detect[x[:,0],x[:,1]]
                sorted_nod, idx = torch.sort(torch.squeeze(t))
                if idx.numel():
                    self.detect_offset = torch.mean(sorted_nod[int((1-Pf)*len(sorted_nod))])
                detect = detect - self.detect_offset
                angle_est = self.rad_angle_est(c_k,targets)
            else:
                detect = detect - self.detect_offset
                detect = torch.sigmoid(detect)
                angle_est = self.rad_angle_est(c_k)
        elif self.encoding=="onehot":
            #offs = torch.arange(0,self.targetnum+1).to(device)
            #offs[0] = -torch.sum(torch.arange(0,self.targetnum+1))
            offs = -torch.ones(self.targetnum+1,device=device)/self.targetnum
            offs[0] = 1
            if targets!=None:
                detect = torch.softmax(detect,1)
                num_targets = torch.sum(targets).to(device)
                m_targs = c_k.size()[0]*self.targetnum-num_targets
                t_help_f = torch.zeros_like(detect)
                target_labels = torch.sum(targets,1)
                for l in range(detect.size()[0]):
                    if target_labels[l]<self.targetnum: 
                        t_help_f[l,int(target_labels[l]+1):] = torch.arange(1, int(self.targetnum-target_labels[l]+1))
                #t_help_argmax = torch.argmax(detect,axis=1)
                #t_NN_h = torch.nn.functional.one_hot(t_help_argmax,self.targetnum+1)
                prob_f_d = 1/m_targs*torch.sum(detect*t_help_f)

                self.detect_offset =  Pf - prob_f_d
                detect = torch.clip(detect + self.detect_offset*offs,0,1)
                angle_est = self.rad_angle_est(c_k)
            else:
                detect = torch.softmax(detect,1)
                detect = torch.clip(detect + self.detect_offset*offs,0,1)
                angle_est = self.rad_angle_est(c_k)
        
        return(detect, angle_est)

class Radar_detect(nn.Module):
    def __init__(self,k,max_target,encoding):
        super(Radar_detect, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.d = torch.prod(k)
        self.targetnum = max_target
        #layers target_detection
        self.fcB1a = nn.Linear(self.d,self.d*2, device=device)
        self.fcB1b = nn.Linear(self.d,self.d*2, device=device)
        self.fcB2 = nn.Linear(self.d*2,self.d*2, device=device)
        self.fcB3 = nn.Linear(self.d*2,self.d, device=device)
        if encoding=='onehot':
            self.fcB4 = nn.Linear(self.d,self.targetnum+1, device=device)
        else:
            self.fcB4 = nn.Linear(self.d,self.targetnum, device=device) # linear output layer, add one for onehot encoding
        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()
    
    def forward(self, c_k):
        detect = self.target_detection(c_k)
        # fix false alarm rate to 0.01 in receiver
        return detect

    def target_detection(self, c_k):
        out = self.activation_function(self.fcB1a(torch.abs(c_k).type(torch.float32))).to(device) + self.activation_function(self.fcB1b(torch.angle(c_k).type(torch.float32))).to(device)
        out = self.activation_function(self.fcB2(out))
        out_2 = self.activation_function(self.fcB3(out))
        outx = (self.activation_function(self.fcB4(out_2)))
        # integrate sigmoid layer into loss function
        return(outx)

class Angle_est(nn.Module):
    def __init__(self,k,max_target=1):
        super(Angle_est, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.d = torch.prod(k)
        self.num_targets = max_target # prevent large network
        t=1
        #layers AoA est
        self.fcA1a = nn.Linear(self.d,self.d*2*t, device=device)
        self.fcA1b = nn.Linear(self.d,self.d*2*t, device=device)
        self.fcA2 = nn.Linear(self.d*2*t,self.d*2*t, device=device)
        self.fcA3 = nn.Linear(self.d*2*t,self.d*t, device=device)
        self.fcA4 = nn.Linear(self.d*t,2*max_target, device=device) # linear output layer 
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def angle_est(self, c_k):
        c_k = c_k / torch.mean(torch.abs(c_k)) # input normalization
        out = self.activation_function(self.fcA1a(torch.real(c_k).type(torch.float32))).to(device)+self.activation_function(self.fcA1b(torch.imag(c_k).type(torch.float32))).to(device)
        out = self.activation_function(self.fcA2(out))
        out_2 = self.activation_function(self.fcA3(out))
        outx = (self.activation_function(self.fcA4(out_2)))
        outx = torch.tanh(outx)*np.pi/2 # now two angles, elevation and azimuth
        out_all = torch.reshape(outx,(-1,2,self.num_targets))
        return(out_all)
    
    def forward(self, c_k, targets=None):
        if targets==None:
            angle = self.angle_est(c_k)
        else:
            targ = torch.nonzero(torch.squeeze(targets))
            angle = torch.zeros((targets.size()[0],2, self.num_targets)).to(device)
            if targ.numel():
                angle[targ[:,0]] = self.angle_est(c_k[torch.squeeze(targ[:,0])])
        return angle

def bceloss_sym(M, my_outputs, mylabels):
    M = M.detach().cpu().numpy()
    loss=torch.zeros(int(np.log2(M)), device=device)
    binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,M,dtype='uint8')), (-1,8)),dtype=torch.float32, device=device)
    binaries = binaries[:,int(8-np.log2(M)):]
    b_labels = binaries[mylabels].int()
    # calculate bitwise estimates
    bitnum = int(np.log2(M))

    for bit in range(bitnum):
        pos_0 = torch.where(binaries[:,bit]==0)[0]
        pos_1 = torch.where(binaries[:,bit]==1)[0]
        est_0 = torch.sum(torch.index_select(my_outputs,1,pos_0), axis=1)
        est_1 = torch.sum(torch.index_select(my_outputs,1,pos_1), axis=1)
        llr = torch.log((est_0+1e-12)/(est_1+1e-12)+1e-12)
        loss[bit]=1/(len(mylabels))*torch.sum(torch.log2(torch.exp((2*b_labels[:,bit]-1)*llr)+1+1e-12), axis=0)
    return torch.sum(loss)
