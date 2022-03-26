import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import save_model
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, output_dir='./', learning_rate=3e-4, betas=(0.9, 0.999), grad_norm_clip=1.0,
                 multi_gpu=False, print_step_size=100, save_checkpoint=1000):
        self.model = model

        self.learning_rate = learning_rate
        self.betas = betas
        self.grad_norm_clip = grad_norm_clip

        self.output_dir = output_dir
        self.writer = SummaryWriter(self.output_dir)
        self.print_step_size = print_step_size
        self.save_checkpoint = save_checkpoint

        self.LB_list = []
        self.loss_params_list = []
        self.KLD_latent_list = []
        self.reconstruct_list = []

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            if multi_gpu:
                self.model = torch.nn.DataParallel(self.model).to(self.device)

    def _save_checkpoint(self, base_dir, info, valid_loss):
        """save checkpoint during training. Format: model_{info}_{valid_loss}"""
        base_name = f'model_{info}_{valid_loss:.3f}'
        logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)

    def train(self, data_obj, n_steps=300000, restart_step=0, batch_size=100):
        model = self.model
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(self.learning_rate, self.betas)

        data_size = data_obj.x_train.shape[0]
        neff = data_obj.Neff
        sampling_probs = data_obj.weights / neff

        step, loss = 0, float('inf')
        while (step + restart_step) < n_steps:
            step += 1
            # Sample a batch from dataset
            batch_index = np.random.choice(data_size, batch_size, p=sampling_probs).tolist()
            data = torch.Tensor(data_obj.x_train[batch_index]).to(self.device)

            loss = self._run_step(data, model, optimizer, neff)
            if step % self.print_step_size == 0:
                index_range = np.arange(step-self.print_step_size, step)

                logger.info(f"Step={step + restart_step}, "
                            f"loss={np.mean(np.array(self.LB_list)[index_range])}, "
                            f"Params={np.mean(np.array(self.loss_params_list)[index_range])}, "
                            f"Latent={np.mean(np.array(self.KLD_latent_list)[index_range])}, "
                            f"Reconstruct={np.mean(np.array(self.reconstruct_list)[index_range])}")
            self.writer.add_scalar('loss', loss, step + 1)  # rewrite if having test dataset

            if self.save_checkpoint and step % self.save_checkpoint == 0:
                self._save_checkpoint(self.output_dir, str(step), loss)

        self._save_checkpoint(self.output_dir, 'final', loss)

    def _run_step(self, x, model, optimizer, nef):
        model.train()
        x = x.to(self.device)
        with torch.set_grad_enabled(True):
            x_reconstructed, logp_xz, z_KL_div, x_KL_div = model(x)
            loss = torch.mean(logp_xz + z_KL_div) + (x_KL_div/nef)

            self.LB_list.append(loss.cpu().detach().numpy())
            self.loss_params_list.append(x_KL_div.cpu().detach().numpy()/nef)
            self.KLD_latent_list.append(z_KL_div.mean().cpu().detach().numpy())
            self.reconstruct_list.append(logp_xz.mean().cpu().detach().numpy())

            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
        optimizer.step()

        return loss