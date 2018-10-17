import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim

        ### to be initialized and passed on from args
        # ngf_gate
        # ngres_gate
        # dropout_gate
        # gate_affine
        # might have to change the embed_size(too large at the moment)
        self.ngres=12
        self.gate_affine = gate_affine
        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim, 16*nf*s0*s0)

        resnet = []
        resnet += [GatedResnetBlock(16*nf, 16*nf)]
        resnet += [GatedResnetBlock(16*nf, 16*nf)]

        resnet += [GatedResnetBlock(16*nf, 16*nf)]
        resnet += [GatedResnetBlock(16*nf, 16*nf)]

        resnet += [GatedResnetBlock(16*nf, 8*nf)]
        resnet += [GatedResnetBlock(8*nf, 8*nf)]

        resnet += [GatedResnetBlock(8*nf, 4*nf)]
        resnet += [GatedResnetBlock(4*nf, 4*nf)]

        resnet += [GatedResnetBlock(4*nf, 2*nf)]
        resnet += [GatedResnetBlock(2*nf, 2*nf)]

        resnet += [GatedResnetBlock(2*nf, 1*nf)]
        resnet += [GatedResnetBlock(1*nf, 1*nf)]

        self.resnet = nn.Sequential(*resnet)

        self.conv_img = spectral_norm(nn.Conv2d(nf, 3, 3, padding=1))

        gate_block =[]
        #gate_block+=[ nn.Linear(opt.nsalient ,opt.ngf_gate)]
        gate_block+=[ Reshape( -1, 1 ,embed_size)  ]
        gate_block+=[ nn.Conv1d(1,ngf_gate,kernel_size=3,stride=1,padding=1)  ]
        #gate_block+=[ nn.InstanceNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        for i in range(ngres_gate):
            gate_block+=[ResBlock1D(ngf_gate,dropout_gate)]
        # state size (opt.batchSize, opt.ngf_gate, opt.nsalient)
        gate_block+=[Reshape(-1,ngf_gate*embed_size)]

        self.gate=nn.Sequential(*gate_block)

        gate_block_mult = []
        gate_block_mult+=[ nn.Linear(ngf_gate*embed_size,self.ngres) ]
        gate_block_mult+= [ nn.Sigmoid()]# [nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate_mult = nn.Sequential(*gate_block_mult)
        if self.gate_affine:
            gate_block_add = []
            gate_block_add+=[ nn.Linear(ngf_gate*embed_size,self.ngres) ]
            gate_block_add+=[nn.Tanh()]
            self.gate_add=nn.Sequential(*gate_block_add)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        output_gate = self.gate(yembed)
        output_gate_mult = self.gate_mult(output_gate)
        if self.opt.gate_affine:
            output_gate_add = self.gate_add(output_gate)

        #yz = torch.cat([z, yembed], dim=1)
        out = self.fc(z)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        #out = self.resnet_0_0(out)
        #out = self.resnet_0_1(out)

        #out = F.upsample(out, scale_factor=2)
        #out = self.resnet_1_0(out)
        #out = self.resnet_1_1(out)

        #out = F.upsample(out, scale_factor=2)
        #out = self.resnet_2_0(out)
        #out = self.resnet_2_1(out)

        #out = F.upsample(out, scale_factor=2)
        #out = self.resnet_3_0(out)
        #out = self.resnet_3_1(out)

        #out = F.upsample(out, scale_factor=2)
        #out = self.resnet_4_0(out)
        #out = self.resnet_4_1(out)

        #out = F.upsample(out, scale_factor=2)
        #out = self.resnet_5_0(out)
        #out = self.resnet_5_1(out)

        for i in range(self.ngres):
            if i%3==2:
                out = F.upsample(out, scale_factor=2)
            alpha=output_gate_mult[:,i]
            alpha=alpha.resize(batch_size,1,1,1)
            if self.gate_affine:
                beta=output_gate_add[:,i]
                beta=beta.resize(batch_size,1,1,1)
                out = self.resnet[i](out,alpha,beta)
            else:
                out = self.resnet[i](out,alpha,beta)

        out = self.conv_img(actvn(out))
        out = F.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        ny = nlabels
        ### to be initialized and passed on from args
        # ndf_gate
        # ndres_gate
        # dropout_gate
        # gate_affine
        # might have to change the embed_size(too large at the moment)
        self.ndres=12
        self.gate_affine=gate_affine
        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.conv_img = spectral_norm(nn.Conv2d(3, 1*nf, 3, padding=1))

        resnet = []
        resnet += GatedResnetBlock(1*nf, 1*nf)
        resnet += GatedResnetBlock(1*nf, 2*nf)

        resnet += GatedResnetBlock(2*nf, 2*nf)
        resnet += GatedResnetBlock(2*nf, 4*nf)

        resnet += GatedResnetBlock(4*nf, 4*nf)
        resnet += GatedResnetBlock(4*nf, 8*nf)

        resnet += GatedResnetBlock(8*nf, 8*nf)
        resnet += GatedResnetBlock(8*nf, 16*nf)

        resnet += GatedResnetBlock(16*nf, 16*nf)
        resnet += GatedResnetBlock(16*nf, 16*nf)

        resnet += GatedResnetBlock(16*nf, 16*nf)
        resnet += GatedResnetBlock(16*nf, 16*nf)

        self.resnet=nn.Sequential(*resnet)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)

        gate_block =[]
        gate_block+=[ Reshape( -1, 1 ,embed_size)  ]
        gate_block+=[ nn.Conv1d(1,ndf_gate,kernel_size=3,stride=1,padding=1)  ]


        #gate_block+=[ nn.Linear(opt.nsalient ,opt.ndf_gate)]
        #gate_block+=[ nn.InstanceNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        for i in range(ndres_gate):
            gate_block+=[ResBlock1D(ndf_gate,dropout_gate)]
        # state_size (opt.batchSize,opt.ndf_gate,opt.nsalient)
        gate_block+= [Reshape(-1,ndf_gate*embed_size)]

        self.gate = nn.Sequential(*gate_block)

        gate_block_mult=[]
        gate_block_mult+=[ nn.Linear(ndf_gate*embed_size,self.ndres) ]
        gate_block_mult+= [nn.Sigmoid()] #[nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate_mult = nn.Sequential(*gate_block_mult)

        if self.gate_affine:
            gate_block_add = []
            gate_block_add+=[ nn.Linear(ndf_gate*embed_size,self.ndres) ]
            gate_block_add+=[nn.Tanh()]
            self.gate_add=nn.Sequential(*gate_block_add)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        output_gate = self.gate(yembed)
        output_gate_mult = self.gate_mult(output_gate)
        if self.opt.gate_affine:
            output_gate_add = self.gate_add(output_gate)

        out = self.conv_img(x)

        #out = self.resnet_0_0(out)
        #out = self.resnet_0_1(out)

        #out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #out = self.resnet_1_0(out)
        #out = self.resnet_1_1(out)

        #out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #out = self.resnet_2_0(out)
        #out = self.resnet_2_1(out)

        #out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #out = self.resnet_3_0(out)
        #out = self.resnet_3_1(out)

        #out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #out = self.resnet_4_0(out)
        #out = self.resnet_4_1(out)

        #out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #out = self.resnet_5_0(out)
        #out = self.resnet_5_1(out)

        for i in range(self.ndres):
            if i%3==2:
                out = F.avg_pool2d(out, 3, stride=2, padding=1)
            alpha=output_gate_mult[:,i]
            alpha=alpha.resize(batch_size,1,1,1)
            if self.gate_affine:
                beta=output_gate_add[:,i]
                beta=beta.resize(batch_size,1,1,1)
                out = self.resnet[i](out,alpha,beta)
            else:
                out = self.resnet[i](out,alpha,beta)

        out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))
        if self.learned_shortcut:
            self.conv_s = spectral_norm( nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False))


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class GatedResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))
        if self.learned_shortcut:
            self.conv_s = spectral_norm( nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False))


    def forward(self, x,alpha=1.0,beta=0.0):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))

        if type(alpha)!=float:
            alpha=alpha.expand_as(x_s)
        if type(beta)!=float:
            beta=beta.expand_as(x_s)
        out = x_s + alpha*dx + beta   #x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s
def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
