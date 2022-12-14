import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np
import wandb

from src.exporter import PpujikPpujik
from src.models.deepspeech import Jangnan
from src.models.pix2pix import PatchDisc, GANLoss, SimpleDisc
from src.models.generator import CNNGenerator, FCGenerator, FCAttentionGenerator
from src.models.full_deepspeech import DeepSpeech
from src.models.cnn import resnet34
from src.utils import CosineAnnealingWarmUpRestarts


class S2BModel(pl.LightningModule):
    def __init__(self,
                 lr,
                 fc1_dim,
                 fc2_dim,
                 num_classes,
                 lambda_G = 100,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.num_classes = num_classes
        self.lr = lr
        self.lambda_G = lambda_G
        self.save_name = save_name
        self.save_hyperparameters()
        
        
        self.encoder = DeepSpeech.load_model('/shared/air/shared/youngkim/mediazen/pretrained/librispeech_pretrained_v2.pth')
        self.speech_fc = nn.Linear(29, self.num_classes)

        # self.net_G = FCGenerator(fc1_dim, fc2_dim)
        self.net_G = CNNGenerator(frame_window=16, attention_window=8, num_classes=self.num_classes)
        self.criterion_MSE = nn.MSELoss(reduction='sum')

        self.net_D = PatchDisc(in_channels=2)
        self.criterion_GAN = GANLoss()


    def forward(self, x, x_length, y, y_length):
        enc_out, x_length = self.encoder(x, x_length)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        # speech feature for net_D
        speech_features = speech_features.permute(0, 2, 1) # B, T, C
        feature_D = self.speech_fc(speech_features) # B, T, num_classes

        return pred_blendshape, feature_D

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        masked_out = masked_out.unsqueeze(1)
        chopped_y = chopped_y.unsqueeze(1)

        return masked_out, chopped_y # B, 1, T, num_classes

    # def masking_features(self, features, y_length, masked_out):
    #     ones_list = [torch.ones(length, 1024) for length in y_length]
    #     length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

    #     chopped_features = features[:, :max(y_length), :]
    #     masked_features = chopped_features * length_mask
        
    #     B, T, C = masked_features.shape
    #     F = masked_out.shape[-1]
    #     masked_features = masked_features.view((B, T, int(C/F), F)).permute(0, 2, 1, 3)

    #     return masked_features # B * C/F * T * F


    def forward_D(self, masked_features, masked_out, chopped_y, y_length):
        # Fake; stop backprop to the generator by detaching fake
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False, y_length)
        # Real
        real = torch.cat((masked_features, chopped_y), 1)
        pred_real = self.net_D(real)
        loss_D_real = self.criterion_GAN(pred_real, True, y_length)
        # Combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D_fake, loss_D_real, loss_D

    def forward_G(self, masked_features, masked_out, chopped_y, y_length):
        # First, G(x) should fake the discriminator
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake)
        loss_G_GAN = self.criterion_GAN(pred_fake, True, y_length)
        # Second, G(x) = masked_out
        loss_G_MSE = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.num_classes
        loss_G_MSE = loss_G_MSE / element_num
        loss_G = loss_G_GAN + (self.lambda_G * loss_G_MSE)
        return loss_G_GAN, loss_G_MSE, loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_length, y, y_length = batch
        out, features_D = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        masked_features, _ = self.masking_preds(features_D, y, y_length)

        # train net_D
        if optimizer_idx == 1:
            loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_D}

        # train generator
        elif optimizer_idx == 0:
            loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_G}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out, features_D = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        masked_features, _ = self.masking_preds(features_D, y, y_length)

        # net_D
        loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_D_fake', loss_D_fake, on_step=True, on_epoch=True)
        self.log('v_loss_D_real', loss_D_real, on_step=True, on_epoch=True)
        self.log('v_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)

        # net_G
        loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_G_GAN', loss_G_GAN, on_step=True, on_epoch=True)
        self.log('v_loss_G_MSE', loss_G_MSE, prog_bar=True, on_step=True, on_epoch=True)
        self.log('v_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
        
        # return {'loss_D': loss_D, 'loss_G': loss_G}
            

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr',cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'/shared/air/shared/youngkim/mediazen/csv_out/ggeoggleggeoggle/{self.save_name}', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'/shared/air/shared/youngkim/mediazen/csv_out/banjilbanjil/{self.save_name}', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out, _ = self(x, x_length, y, y_length)

        y_length, timecodes, out = y_length.cpu(), timecodes.cpu(), out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)


    def configure_optimizers(self):
        g_params = list(self.encoder.parameters()) + list(self.net_G.parameters())
        opt_g = torch.optim.Adam(g_params, lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # scheduler = {
        #     "scheduler": CosineAnnealingLR(optimizer, **{"T_0": 1, "T_mult": 2, "eta_min": 1e-07}),
        #     "interval": "epoch",
        # }
        # return [optimizer], [scheduler]
        return [opt_g, opt_d], []


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features


class GANFCGenSimpleDisc(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 fc1_dim,
                 fc2_dim,
                 num_classes,
                 lambda_G = 100,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = DeepSpeech.load_model(deepspeech_model_path)

        self.net_G = FCGenerator(fc1_dim, fc2_dim)
        self.criterion_MSE = nn.MSELoss(reduction='sum')

        self.net_D = SimpleDisc(in_channels=1)
        self.criterion_GAN = GANLoss()


    def forward(self, x, x_length, y, y_length):
        enc_out, x_length = self.encoder(x, x_length)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T
        speech_features = speech_features.permute(0, 2, 1).contiguous() # B, T, C

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        return pred_blendshape


    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.hparams.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        masked_out = masked_out.unsqueeze(1)
        chopped_y = chopped_y.unsqueeze(1)

        return masked_out, chopped_y # B, 1, T, num_classes


    def forward_D(self, masked_out, chopped_y, y_length):
        # Fake; stop backprop to the generator by detaching fake
        fake = masked_out
        pred_fake = self.net_D(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False, y_length)
        # Real
        real = chopped_y
        pred_real = self.net_D(real)
        loss_D_real = self.criterion_GAN(pred_real, True, y_length)
        # Combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D_fake, loss_D_real, loss_D

    def forward_G(self, masked_out, chopped_y, y_length):
        # First, G(x) should fake the discriminator
        fake = masked_out
        pred_fake = self.net_D(fake)
        loss_G_GAN = self.criterion_GAN(pred_fake, True, y_length)
        # Second, G(x) = masked_out
        loss_G_MSE = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss_G_MSE = loss_G_MSE / element_num
        loss_G = loss_G_GAN + (self.hparams.lambda_G * loss_G_MSE)
        return loss_G_GAN, loss_G_MSE, loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        # train net_D
        if optimizer_idx == 1:
            loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_out, chopped_y, y_length)
            self.log('t_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_D}

        # train generator
        elif optimizer_idx == 0:
            loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_out, chopped_y, y_length)
            self.log('t_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_G}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        # net_D
        loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_out, chopped_y, y_length)
        self.log('v_loss_D_fake', loss_D_fake, on_step=True, on_epoch=True)
        self.log('v_loss_D_real', loss_D_real, on_step=True, on_epoch=True)
        self.log('v_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)

        # net_G
        loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_out, chopped_y, y_length)
        self.log('v_loss_G_GAN', loss_G_GAN, on_step=True, on_epoch=True)
        self.log('v_loss_G_MSE', loss_G_MSE, prog_bar=True, on_step=True, on_epoch=True)
        self.log('v_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
        
        # return {'loss_D': loss_D, 'loss_G': loss_G}
            

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr',cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)


    def configure_optimizers(self):
        g_params = list(self.encoder.parameters()) + list(self.net_G.parameters())
        opt_g = torch.optim.AdamW(g_params, lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.net_D.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        # scheduler = {
        #     "scheduler": CosineAnnealingLR(optimizer, **{"T_0": 1, "T_mult": 2, "eta_min": 1e-07}),
        #     "interval": "epoch",
        # }
        # return [optimizer], [scheduler]
        return [opt_g, opt_d], []


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features


class GANCNNGenPatchDisc(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 num_classes,
                 lambda_G = 100,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.lambda_G = lambda_G
        self.save_name = save_name
        self.save_hyperparameters()
        
        
        self.encoder = DeepSpeech.load_model(deepspeech_model_path)
        self.speech_fc = nn.Linear(29, self.num_classes)

        self.net_G = CNNGenerator(frame_window=16, attention_window=8, num_classes=self.num_classes)
        self.criterion_MSE = nn.MSELoss(reduction='sum')

        self.net_D = PatchDisc(in_channels=2)
        self.criterion_GAN = GANLoss()


    def forward(self, x, x_length, y, y_length):
        enc_out, x_length = self.encoder(x, x_length)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        # speech feature for net_D
        speech_features = speech_features.permute(0, 2, 1) # B, T, C
        feature_D = self.speech_fc(speech_features) # B, T, num_classes

        return pred_blendshape, feature_D

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        masked_out = masked_out.unsqueeze(1)
        chopped_y = chopped_y.unsqueeze(1)

        return masked_out, chopped_y # B, 1, T, num_classes


    def forward_D(self, masked_features, masked_out, chopped_y, y_length):
        # Fake; stop backprop to the generator by detaching fake
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False, y_length)
        # Real
        real = torch.cat((masked_features, chopped_y), 1)
        pred_real = self.net_D(real)
        loss_D_real = self.criterion_GAN(pred_real, True, y_length)
        # Combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D_fake, loss_D_real, loss_D

    def forward_G(self, masked_features, masked_out, chopped_y, y_length):
        # First, G(x) should fake the discriminator
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake)
        loss_G_GAN = self.criterion_GAN(pred_fake, True, y_length)
        # Second, G(x) = masked_out
        loss_G_MSE = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.num_classes
        loss_G_MSE = loss_G_MSE / element_num
        loss_G = loss_G_GAN + (self.lambda_G * loss_G_MSE)
        return loss_G_GAN, loss_G_MSE, loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_length, y, y_length = batch
        out, features_D = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        masked_features, _ = self.masking_preds(features_D, y, y_length)

        # train net_D
        if optimizer_idx == 1:
            loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_D}

        # train generator
        elif optimizer_idx == 0:
            loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_G}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out, features_D = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        masked_features, _ = self.masking_preds(features_D, y, y_length)

        # net_D
        loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_D_fake', loss_D_fake, on_step=True, on_epoch=True)
        self.log('v_loss_D_real', loss_D_real, on_step=True, on_epoch=True)
        self.log('v_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)

        # net_G
        loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_G_GAN', loss_G_GAN, on_step=True, on_epoch=True)
        self.log('v_loss_G_MSE', loss_G_MSE, prog_bar=True, on_step=True, on_epoch=True)
        self.log('v_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
        
        # return {'loss_D': loss_D, 'loss_G': loss_G}
            

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr',cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out, _ = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)


    def configure_optimizers(self):
        g_params = list(self.encoder.parameters()) + list(self.net_G.parameters())
        opt_g = torch.optim.Adam(g_params, lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features


class GANCNNGenPatchDiscFeatureCNN(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 num_classes,
                 lambda_G = 100,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.lambda_G = lambda_G
        self.save_name = save_name
        self.save_hyperparameters()
        
        
        self.encoder = DeepSpeech.load_model(deepspeech_model_path)
        self.speech_cnn = resnet34()

        self.net_G = CNNGenerator(frame_window=16, attention_window=8, num_classes=self.num_classes)
        self.criterion_MSE = nn.MSELoss(reduction='sum')

        self.net_D = PatchDisc(in_channels=65)
        self.criterion_GAN = GANLoss()


    def forward(self, x, x_length, y, y_length):
        enc_out, rnn_out, x_length = self.encoder(x, x_length, return_both=True)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T
        rnn_out = rnn_out.permute(1, 2, 0).contiguous()
        rnn_out_intp = self.interpolate_features(rnn_out, x_length, y_length)
        rnn_out_intp = rnn_out_intp.unsqeeze(1)

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        # speech feature for net_D
        feature_D = self.speech_cnn(rnn_out_intp, y_length) # B, 64, 16, T
        feature_D = feature_D.permute(0, 1, 3, 2).contiguous() # B, 64, T, 16

        return pred_blendshape, feature_D

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        masked_out = masked_out.unsqueeze(1)
        chopped_y = chopped_y.unsqueeze(1)

        return masked_out, chopped_y # B, 1, T, num_classes


    def forward_D(self, masked_features, masked_out, chopped_y, y_length):
        # Fake; stop backprop to the generator by detaching fake
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False, y_length)
        # Real
        real = torch.cat((masked_features, chopped_y), 1)
        pred_real = self.net_D(real)
        loss_D_real = self.criterion_GAN(pred_real, True, y_length)
        # Combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D_fake, loss_D_real, loss_D

    def forward_G(self, masked_features, masked_out, chopped_y, y_length):
        # First, G(x) should fake the discriminator
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake)
        loss_G_GAN = self.criterion_GAN(pred_fake, True, y_length)
        # Second, G(x) = masked_out
        loss_G_MSE = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.num_classes
        loss_G_MSE = loss_G_MSE / element_num
        loss_G = loss_G_GAN + (self.lambda_G * loss_G_MSE)
        return loss_G_GAN, loss_G_MSE, loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_length, y, y_length = batch
        out, masked_features = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        
        # train net_D
        if optimizer_idx == 1:
            loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_D}

        # train generator
        elif optimizer_idx == 0:
            loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_G}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out, masked_features = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        
        # net_D
        loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_D_fake', loss_D_fake, on_step=True, on_epoch=True)
        self.log('v_loss_D_real', loss_D_real, on_step=True, on_epoch=True)
        self.log('v_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)

        # net_G
        loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_G_GAN', loss_G_GAN, on_step=True, on_epoch=True)
        self.log('v_loss_G_MSE', loss_G_MSE, prog_bar=True, on_step=True, on_epoch=True)
        self.log('v_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
        
        # return {'loss_D': loss_D, 'loss_G': loss_G}
            

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr',cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out, _ = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)


    def configure_optimizers(self):
        g_params = list(self.encoder.parameters()) + list(self.net_G.parameters())
        opt_g = torch.optim.Adam(g_params, lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.net_D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features


class GANFCGenPatchDiscFeatureCNN(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 fc1_dim, 
                 fc2_dim,
                 num_classes,
                 lambda_G = 100,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        
        self.encoder = DeepSpeech.load_model(deepspeech_model_path)
        self.speech_cnn = resnet34()

        self.net_G = FCGenerator(fc1_dim, fc2_dim)
        self.criterion_MSE = nn.MSELoss(reduction='sum')

        self.net_D = PatchDisc(in_channels=65)
        self.criterion_GAN = GANLoss()


    def forward(self, x, x_length, y, y_length):
        enc_out, rnn_out, x_length = self.encoder(x, x_length, return_both=True)
        
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T
        speech_features = speech_features.permute(0, 2, 1).contiguous() # B, T, C
        
        rnn_out = rnn_out.permute(1, 2, 0).contiguous()
        rnn_out_intp = self.interpolate_features(rnn_out, x_length, y_length)
        rnn_out_intp = rnn_out_intp.unsqueeze(1)

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        # speech feature for net_D
        feature_D = self.speech_cnn(rnn_out_intp, y_length) # B, 64, 16, T
        feature_D = feature_D.permute(0, 1, 3, 2).contiguous() # B, 64, T, 16

        return pred_blendshape, feature_D

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.hparams.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        masked_out = masked_out.unsqueeze(1)
        chopped_y = chopped_y.unsqueeze(1)

        return masked_out, chopped_y # B, 1, T, num_classes


    def forward_D(self, masked_features, masked_out, chopped_y, y_length):
        # Fake; stop backprop to the generator by detaching fake
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False, y_length)
        # Real
        real = torch.cat((masked_features, chopped_y), 1)
        pred_real = self.net_D(real)
        loss_D_real = self.criterion_GAN(pred_real, True, y_length)
        # Combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D_fake, loss_D_real, loss_D

    def forward_G(self, masked_features, masked_out, chopped_y, y_length):
        # First, G(x) should fake the discriminator
        fake = torch.cat((masked_features, masked_out), 1)
        pred_fake = self.net_D(fake)
        loss_G_GAN = self.criterion_GAN(pred_fake, True, y_length)
        # Second, G(x) = masked_out
        loss_G_MSE = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss_G_MSE = loss_G_MSE / element_num
        loss_G = loss_G_GAN + (self.hparams.lambda_G * loss_G_MSE)
        return loss_G_GAN, loss_G_MSE, loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_length, y, y_length = batch
        out, masked_features = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        
        # train net_D
        if optimizer_idx == 1:
            loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_D}

        # train generator
        elif optimizer_idx == 0:
            loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
            self.log('t_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
            return {'loss': loss_G}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out, masked_features = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)
        
        # net_D
        loss_D_fake, loss_D_real, loss_D = self.forward_D(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_D_fake', loss_D_fake, on_step=True, on_epoch=True)
        self.log('v_loss_D_real', loss_D_real, on_step=True, on_epoch=True)
        self.log('v_loss_D', loss_D, prog_bar=True, on_step=True, on_epoch=True)

        # net_G
        loss_G_GAN, loss_G_MSE, loss_G = self.forward_G(masked_features, masked_out, chopped_y, y_length)
        self.log('v_loss_G_GAN', loss_G_GAN, on_step=True, on_epoch=True)
        self.log('v_loss_G_MSE', loss_G_MSE, prog_bar=True, on_step=True, on_epoch=True)
        self.log('v_loss_G', loss_G, prog_bar=True, on_step=True, on_epoch=True)
        
        # return {'loss_D': loss_D, 'loss_G': loss_G}
            

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr',cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out, _ = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)


    def configure_optimizers(self):
        g_params = list(self.encoder.parameters()) + list(self.net_G.parameters())
        opt_g = torch.optim.Adam(g_params, lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.net_D.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features


class SimpleCNN(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 num_classes,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = DeepSpeech.load_model(deepspeech_model_path)
        self.net_G = CNNGenerator(frame_window=16, attention_window=8, num_classes=num_classes)
        self.criterion_MSE = nn.MSELoss(reduction='sum')
    
    def forward(self, x, x_length, y, y_length):
        # with torch.no_grad():
        enc_out, x_length = self.encoder(x, x_length)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        return pred_blendshape

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.hparams.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        return masked_out, chopped_y # B, T, num_classes

    def training_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        sum_loss = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss = sum_loss / element_num
        self.log("t_loss_G", loss, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        sum_loss = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss = sum_loss / element_num
        self.log("v_loss_G_MSE", loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = {
        #     "scheduler": CosineAnnealingLR(optimizer, **{"T_0": 1, "T_mult": 2, "eta_min": 1e-07}),
        #     "interval": "epoch",
        # }
        # return [optimizer], [scheduler]
        return [optimizer]

    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features

    
class SimpleFC(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 fc1_dim,
                 fc2_dim,
                 num_classes,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = DeepSpeech.load_model(deepspeech_model_path)
        self.net_G = FCGenerator(fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.criterion_MSE = nn.MSELoss(reduction='sum')
    
    def forward(self, x, x_length, y, y_length):
        # with torch.no_grad():
        enc_out, x_length = self.encoder(x, x_length, return_rnn_out=True)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T
        speech_features = speech_features.permute(0, 2, 1).contiguous() # B, T, C

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        return pred_blendshape

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.hparams.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        return masked_out, chopped_y # B, T, num_classes

    def training_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        sum_loss = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss = sum_loss / element_num
        self.log("t_loss_G", loss, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        sum_loss = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss = sum_loss / element_num
        self.log("v_loss_G_MSE", loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)

    def predict_step(self, batch, batch_idx):
        x, x_length, f_name = batch
        f_name = f_name[0][:-4]
        out = self(x, x_length, None, None)
        out = out.cpu()

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{f_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{f_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))
        
        self.jururuk.save_to_csv_predict(f_name, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.save_to_csv_predict(f_name, self.trainer.datamodule.blendshape_columns, out)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = {
        #     "scheduler": CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.0005, T_up=2, gamma=0.5),
        #     "interval": "epoch",
        # }
        # return [optimizer], [scheduler]
        return [optimizer]


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(
                features, 
                torch.max(output_len),
                mode='linear',
                align_corners=True)
            return output_features



class SimpleFCAttention(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 deepspeech_model_path,
                 fc1_dim,
                 fc2_dim,
                 num_classes,
                 attention_window=8,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = DeepSpeech.load_model(deepspeech_model_path)
        self.net_G = FCAttentionGenerator(fc1_dim=fc1_dim, fc2_dim=fc2_dim, attention_window=attention_window)
        self.criterion_MSE = nn.MSELoss(reduction='sum')
    
    def forward(self, x, x_length, y, y_length):
        # with torch.no_grad():
        enc_out, x_length = self.encoder(x, x_length, return_rnn_out=True)
        enc_out = enc_out.permute(1, 2, 0).contiguous()
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T
        speech_features = speech_features.permute(0, 2, 1).contiguous() # B, T, C

        # net_G
        pred_blendshape = self.net_G(speech_features) # B, T, num_classes

        return pred_blendshape

    def masking_preds(self, out, y, y_length):
        ones_list = [torch.ones(length, self.hparams.num_classes) for length in y_length]
        length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

        chopped_out = out[:, :max(y_length), :]
        chopped_y = y[:, :max(y_length), :]
        masked_out = chopped_out * length_mask

        return masked_out, chopped_y # B, T, num_classes

    def training_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        sum_loss = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss = sum_loss / element_num
        self.log("t_loss_G", loss, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, x_length, y, y_length = batch
        out = self(x, x_length, y, y_length)

        masked_out, chopped_y = self.masking_preds(out, y, y_length)

        sum_loss = self.criterion_MSE(masked_out, chopped_y)
        element_num = torch.sum(y_length) * self.hparams.num_classes
        loss = sum_loss / element_num
        self.log("v_loss_G_MSE", loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{self.hparams.save_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes, f_names = batch
        out = self(x, x_length, y, y_length)

        chopped_out = out[:, :max(y_length), :]
        chopped_timecodes = timecodes[:, :max(y_length)]

        y_length, timecodes, out = y_length.cpu(), chopped_timecodes.cpu(), chopped_out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, f_names, timecodes, self.trainer.datamodule.blendshape_columns, out)

    def predict_step(self, batch, batch_idx):
        x, x_length, f_name = batch
        f_name = f_name[0][:-4]
        out = self(x, x_length, None, None)
        out = out.cpu()

        self.jururuk = PpujikPpujik(f'{self.hparams.csv_out_dir}/{f_name}/ggeoggleggeoggle', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'{self.hparams.csv_out_dir}/{f_name}/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))
        
        self.jururuk.save_to_csv_predict(f_name, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.save_to_csv_predict(f_name, self.trainer.datamodule.blendshape_columns, out)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = {
        #     "scheduler": CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.0005, T_up=2, gamma=0.5),
        #     "interval": "epoch",
        # }
        # return [optimizer], [scheduler]
        return [optimizer]


    def interpolate_features(self, features, f_len, b_len=None, input_rate=50, output_rate=60):
        if b_len is not None:
            batch_size = features.shape[0]
            num_features = features.shape[1]
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)), device=self.device)
            
            for b in range(batch_size):
                interp = nn.functional.interpolate(
                    features[b:b+1,:,:f_len[b]],
                    b_len[b],
                    mode='linear',
                    align_corners=True
                    )
                output_features[b:b+1,:,:b_len[b]] = interp
            return output_features
        else:
            output_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(
                features, 
                torch.max(output_len),
                mode='linear',
                align_corners=True)
            return output_features