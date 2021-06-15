from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import os


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.logdir = config.logdir
        self.warmup_path = config.warmup_path


        # Build the model and tensorboard.
        self.current_step = self.build_model()

        self.writer = SummaryWriter(self.logdir)

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)  
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)

        current_step = 0
        if os.path.exists(self.warmup_path):
            print(f'\n【Warmup model from {self.warmup_path}】\n')
            checkpoint = torch.load(self.warmup_path)
            model_dict = checkpoint['state_dict']
            self.G.load_state_dict(model_dict)
            current_step = checkpoint['global_step'] + 1 
            self.g_optimizer = checkpoint['optimizer']
        
        print(f'\n【current model step is {current_step}】\n')
        self.G.to(self.device)
        
        return current_step
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.current_step , self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)

            x_identic = x_identic.squeeze()
            x_identic_psnt = x_identic_psnt.squeeze()
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                self.writer.add_scalar("loss/org_rec", g_loss_id.item(), i + 1)
                self.writer.add_scalar("loss/pst_rec", g_loss_id_psnt.item(), i + 1)
                self.writer.add_scalar("loss/content", g_loss_cd.item(), i + 1)
                self.writer.add_image("image/ori_mel", x_real[:1,:,:].permute(0,2,1), i + 1)
                self.writer.add_image("image/pre_mel", x_identic[:1,:,:].permute(0,2,1), i + 1)
                self.writer.add_image("image/pre_pos_mel", x_identic_psnt[:1,:,:].permute(0,2,1), i + 1)
                
            if (i+1) % self.save_step == 0:
                checkpoint_file_name = os.path.join(self.logdir, 'step_%01dK.pth'%(i / 1000))
                print("saving the checkpoint file '%s'..." % checkpoint_file_name)
                checkpoint = {
                    'global_step' : i + 1,
                    'state_dict' :  self.G.state_dict(),
                    'optimizer' : self.g_optimizer
                }
                torch.save(checkpoint, checkpoint_file_name)
                torch.save(checkpoint, os.path.join(self.logdir, 'model.pth'))
                del checkpoint
    
    

    
