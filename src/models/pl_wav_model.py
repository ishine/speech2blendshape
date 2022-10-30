import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np
import fairseq
from fairseq.models.wav2vec import Wav2Vec2Model
import wandb

from src.exporter import PpujikPpujik
from src.models.deepspeech import Jangnan
from src.models.pix2pix import PatchDisc, GANLoss, SimpleDisc
from src.models.generator import CNNGenerator, FCGenerator
from src.models.full_deepspeech import DeepSpeech
from src.models.cnn import resnet34
from src.utils import CosineAnnealingWarmUpRestarts

    
class SimpleFC(pl.LightningModule):
    def __init__(self,
                 csv_out_dir,
                 lr,
                 fc1_dim,
                 fc2_dim,
                 num_classes,
                 save_name = 'baseline'
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.net_G = FCGenerator(fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.criterion_MSE = nn.MSELoss(reduction='sum')
    

    def forward(self, x, x_length, y, y_length):
        # x : B T C
        # enc_out, x_length = self.encoder(x, x_length, return_rnn_out=True)
        enc_out = x.permute(0, 2, 1) # B, C, T
        speech_features = self.interpolate_features(enc_out, x_length, y_length) # B, C, T
        speech_features = speech_features.permute(0, 2, 1) # B, T, C

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
            output_features = nn.functional.interpolate(features, torch.max(output_len))
            return output_features