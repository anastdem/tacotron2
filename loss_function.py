import torch
from torch import nn
from utils import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams, lpips_alex):
        super(Tacotron2Loss, self).__init__()
        self.hparams = hparams
        self.lpips_loss = lpips_alex
        # self.lpips_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores
        # self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()

    def batch_diagonal_guide(self, input_lengths, output_lengths, g=0.2):
        dtype, device = torch.float32, input_lengths.device

        grid_text = torch.arange(input_lengths.max(), dtype=dtype, device=device)
        grid_text = grid_text.view(1, -1) / input_lengths.view(-1, 1)  # (B, T)

        grid_mel = torch.arange(output_lengths.max(), dtype=dtype, device=device)
        grid_mel = grid_mel.view(1, -1) / output_lengths.view(-1, 1)  # (B, M)

        grid = grid_text.unsqueeze(1) - grid_mel.unsqueeze(2)  # (B, M, T)

        # apply text and mel length masks
        grid.transpose(2, 1)[~get_mask_from_lengths(input_lengths)] = 0.
        grid[~get_mask_from_lengths(output_lengths)] = 0.

        W = 1 - torch.exp(-grid ** 2 / (2 * g ** 2))
        return W

    def forward(self, model_output, targets, model_input):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        text_padded, input_lengths, mel_padded, max_len, output_lengths = model_input
        mel_out, mel_out_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        diagonal_guides = self.batch_diagonal_guide(input_lengths, output_lengths, g=self.hparams.diagonal_factor)
        attention_loss = torch.sum(alignments * diagonal_guides)
        active_elements = torch.sum(input_lengths * output_lengths)
        attention_loss = attention_loss / active_elements

        with torch.no_grad():
            lpips_loss = [self.lpips_loss(mel_out_postnet[i, :, :output_lengths[i]], mel_target[i, :, :output_lengths[i]])
                         for i in range(mel_out_postnet.shape[0])]

        return mel_loss + self.hparams.gate_positive_weight * gate_loss + attention_loss * self.hparams.attention_weight, torch.tensor(lpips_loss).mean()
