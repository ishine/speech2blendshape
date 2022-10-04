import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np
import wandb

from src.exporter import PpujikPpujik
from src.models.deepspeech import Jangnan
from src.models.pix2pix import Discriminator, GANLoss
from src.models.generator import CNNGenerator, FCGenerator
from src.models.full_deepspeech import DeepSpeech



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
        
        # self.encoder = Jangnan()
        # self.encoder.load_state_dict(torch.load('/root/mediazen/speech2blendshape/pretrained/jangnan.pt'))
        self.encoder = DeepSpeech.load_model('/root/mediazen/speech2blendshape/pretrained/librispeech_pretrained_v2.pth')
        self.speech_fc = nn.Linear(29, self.num_classes)

        # self.net_G = FCGenerator(fc1_dim, fc2_dim)
        self.net_G = CNNGenerator(frame_window=16, attention_window=8, num_classes=self.num_classes)
        self.criterion_MSE = nn.MSELoss(reduction='sum')

        self.net_D = Discriminator(in_channels=2)
        self.criterion_GAN = GANLoss()


    def forward(self, x, x_length, y, y_length):
        with torch.no_grad():
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

    # def training_epoch_end(self, outputs):
    #     avg_loss_D = torch.stack([x['loss_D'] for x in outputs]).mean()
    #     avg_loss_G = torch.stack([x['loss_G'] for x in outputs]).mean()

    #     self.log('train_loss_D', avg_loss_D)
    #     self.log('train_loss_G', avg_loss_G)

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
        # avg_loss_D = torch.stack([x['val_loss_D'] for x in outputs]).mean()
        # avg_loss_G = torch.stack([x['val_loss_G'] for x in outputs]).mean()

        # self.log('valid_loss_D', avg_loss_D)
        # self.log('valid_loss_G', avg_loss_G)
        
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr',cur_lr)
    
    def test_step(self, batch, batch_idx):

        self.jururuk = PpujikPpujik(f'csv_out/ggeoggleggeoggle/{self.save_name}', PpujikPpujik.ssemssem)
        self.ppujikppujik = PpujikPpujik(f'csv_out/banjilbanjil/{self.save_name}', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

        x, x_length, y, y_length, indices, timecodes = batch
        out, _ = self(x, x_length, y, y_length)

        y_length, indices, timecodes, out = y_length.cpu(), indices.cpu(), timecodes.cpu(), out.cpu()

        self.jururuk.batch_save_to_csvs(y_length, indices, timecodes, self.trainer.datamodule.blendshape_columns, out)
        self.ppujikppujik.batch_save_to_csvs(y_length, indices, timecodes, self.trainer.datamodule.blendshape_columns, out)


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
            output_features = torch.zeros((batch_size, num_features, torch.max(b_len)))
            output_features = output_features.to(self.device)
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
            b_len = torch.ceil(f_len / input_rate * output_rate).to(torch.int32)
            output_features = nn.functional.interpolate(features, torch.max(b_len))
            return output_features



# class GGomYangModel(pl.LightningModule):
#     def __init__(self,
#                  lr,
#                  fc1_dim,
#                  fc2_dim,
#                  num_classes,
#                  ):
#         super().__init__()
#         self.fc1_dim = fc1_dim
#         self.fc2_dim = fc2_dim
#         self.num_classes = num_classes
#         self.lr = lr
#         self.save_hyperparameters()

#         self.sum_criterion = nn.MSELoss(reduction='sum')
        
#         self.jangnan_encoder = Jangnan()
#         self.jangnan_encoder.load_state_dict(torch.load('pretrained/jangnan.pt'))

#         self.jururuk = PpujikPpujik('csv_out/ggeoggleggeoggle', PpujikPpujik.ssemssem)
#         self.ppujikppujik = PpujikPpujik('csv_out/banjilbanjil', PpujikPpujik.ttukttakttukttak_migglemiggle(3,5))

#         fully_connected = nn.Sequential(
#             nn.AdaptiveAvgPool1d(self.fc1_dim),
#             nn.Linear(self.fc1_dim, self.fc2_dim, bias=False),
#             nn.ReLU6(),
#             nn.Dropout(0.2),
#             nn.Linear(self.fc2_dim, self.num_classes, bias=False),
#         )
#         self.fc = nn.Sequential(
#             fully_connected,
#             # nn.Sigmoid()
#             ClipModule(min_threshold=0, max_threshold=1),
#         )
    
#     def forward(self, x, x_length, y, y_length):
#         x = self.jangnan_encoder(x, x_length)
#         x = x.permute(1, 0, 2).contiguous()
#         x = self.interpolate_features(x, 50, 60, output_len= len(y[0]))
#         x = self.fc(x)

#         return x

#     def training_step(self, batch, batch_idx):
#         x, x_length, y, y_length = batch
#         out = self(x, x_length, y, y_length)

#         ones_list = [torch.ones(length, self.num_classes) for length in y_length]
#         length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

#         chopped_out = out[:, :max(y_length), :]
#         chopped_y = y[:, :max(y_length), :]
#         masked_out = chopped_out * length_mask

#         sum_loss = self.sum_criterion(masked_out, chopped_y)
#         element_num = torch.sum(y_length) * self.num_classes
#         loss = sum_loss / element_num

#         # wandb.log({'train_step_loss': loss})

#         return {'loss': loss}

#     def training_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

#         wandb.log({'train_loss': avg_loss}, commit=False)

#     def validation_step(self, batch, batch_idx):
#         x, x_length, y, y_length = batch
#         out = self(x, x_length, y, y_length)

#         ones_list = [torch.ones(length, self.num_classes) for length in y_length]
#         length_mask = torch.nn.utils.rnn.pad_sequence(ones_list, batch_first=True).to(self.device)

#         chopped_out = out[:, :max(y_length), :]
#         chopped_y = y[:, :max(y_length), :]
#         masked_out = chopped_out * length_mask

#         sum_loss = self.sum_criterion(masked_out, chopped_y)
#         element_num = torch.sum(y_length) * self.num_classes
#         loss = sum_loss / element_num
#         self.log("val_loss", loss)
#         return {'val_loss': loss}

#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']

#         wandb.log({'valid_loss': avg_loss, 'lr': cur_lr})
    
#     def test_step(self, batch, batch_idx):
#         x, x_length, y, y_length, indices, timecodes = batch
#         out = self(x, x_length, y, y_length)

#         y_length, indices, timecodes, out = y_length.cpu(), indices.cpu(), timecodes.cpu(), out.cpu()

#         self.jururuk.batch_save_to_csvs(y_length, indices, timecodes, self.trainer.datamodule.blendshape_columns, out)
#         self.ppujikppujik.batch_save_to_csvs(y_length, indices, timecodes, self.trainer.datamodule.blendshape_columns, out)


#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
#         # scheduler = {
#         #     "scheduler": CosineAnnealingLR(optimizer, **{"T_0": 1, "T_mult": 2, "eta_min": 1e-07}),
#         #     "interval": "epoch",
#         # }
#         # return [optimizer], [scheduler]
#         return [optimizer]


#     def interpolate_features(self, features, input_rate, output_rate, output_len=None):
#         batch_size = features.shape[0]
#         num_features = features.shape[2]
#         input_len = features.shape[1]
#         seq_len = input_len / float(input_rate)
#         if output_len is None:
#             output_len = int(seq_len * output_rate)
#         input_timestamps = np.arange(input_len) / float(input_rate)
#         output_timestamps = np.arange(output_len) / float(output_rate)
#         output_features = np.zeros((batch_size, output_len, num_features))
#         features_numpy = features.cpu().data.numpy()
#         for batch in range(batch_size):
#             for feat in range(num_features):
#                 output_features[batch][:, feat] = np.interp(output_timestamps,
#                                                     input_timestamps,
#                                                     features_numpy[batch][:, feat])
#         return torch.from_numpy(output_features).float().to(self.device)
