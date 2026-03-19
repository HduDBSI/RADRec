import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import recall_at_k, ndcg_k, generate_padded_sequences_tensor
from datasets import ENTROPY_GROUP_HIGH, ENTROPY_GROUP_LOW, ENTROPY_GROUP_MID


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, device, args):

        self.args = args
        self.device = device
        self.model = model
        self.seq_model = self.model.seq_model

        self.sim=self.args.sim

        self.train_dataloader = train_dataloader

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.seq_model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HR@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HR@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)


class RADRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, device, args):
        super(RADRecTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, device, args)

    def build_batch_diffusion_condition(self, seq_repr, entropy_groups):
        batch_size = seq_repr.size(0)
        with torch.no_grad():
            empty_mask = entropy_groups == ENTROPY_GROUP_MID
            condition_indices = torch.arange(batch_size, device=seq_repr.device)

            low_indices = torch.nonzero(entropy_groups == ENTROPY_GROUP_LOW, as_tuple=False).flatten()
            high_indices = torch.nonzero(entropy_groups == ENTROPY_GROUP_HIGH, as_tuple=False).flatten()

            if low_indices.numel() > 0 and high_indices.numel() > 0:
                sim_matrix = self.compute_pairwise_similarity(seq_repr.detach(), seq_repr.detach())

                high_to_low = sim_matrix.index_select(0, high_indices).index_select(1, low_indices)
                best_low = low_indices[torch.argmax(high_to_low, dim=1)]
                condition_indices[high_indices] = best_low

                low_to_high = sim_matrix.index_select(0, low_indices).index_select(1, high_indices)
                best_high = high_indices[torch.argmax(low_to_high, dim=1)]
                condition_indices[low_indices] = best_high

        condition_repr = seq_repr.index_select(0, condition_indices)
        return condition_repr, empty_mask

    def compute_pairwise_similarity(self, left_repr, right_repr):
        if self.sim == 'cos':
            left_repr = F.normalize(left_repr, p=2, dim=1)
            right_repr = F.normalize(right_repr, p=2, dim=1)
        return torch.matmul(left_repr, right_repr.transpose(0, 1))

    @staticmethod
    def apply_masked_self_alignment(diffusion_repr, seq_repr, align_mask):
        aligned_repr = diffusion_repr.clone()
        aligned_repr[align_mask] = seq_repr[align_mask]
        return aligned_repr

    def iteration(self, epoch, dataloader, train=True):
        if train:
            self.model.train()
            avg_loss = 0.0

            # print(f"rec dataset length: {len(dataloader)}")
            rec_t_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            # minibatch
            for i, (rec_batch) in rec_t_data_iter:

                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_seq, input_seq_len, target_pos, target, _, entropy_group = rec_batch
                
                x0_gt = self.model.calculate_x0(target) # B x D
                x0_out = self.seq_model(input_seq) # BxLxD
                seq_repr = x0_out[:, -1, :]

                condition_repr, empty_mask = self.build_batch_diffusion_condition(seq_repr, entropy_group)
                
                s = self.model.calculate_s(
                    condition_rep=condition_repr,
                    p=self.args.p,
                    force_empty_mask=empty_mask,
                )

                t = torch.randint(0, self.args.timesteps, (input_seq.size(0), ), device=self.device).long()
                diff_loss, predicted_x_0 = self.calculate_diff_loss(self.model, x0_gt, s, t)
                
                # ablation
                if self.args.diff_weight == 0:
                    sequences_tensor = generate_padded_sequences_tensor(input_seq.size(0), self.args.max_seq_length ,self.args.item_size, input_seq_len)
                    sequences_tensor =sequences_tensor.to(self.device)
                    x0_hat_out = self.seq_model(sequences_tensor)
                    x0_hat_out = x0_hat_out[:,-1,:]
                
                else:
                    x0_hat_out = self.model.sample_from_reverse_process(s) # BxD

                cl_input = self.apply_masked_self_alignment(x0_hat_out, seq_repr, empty_mask)

                cl_loss = self.calculate_cl_loss(cl_input, seq_repr, target_pos[:, -1])

                logits = self.predict_full(seq_repr)  #  Bx|I| 
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos[:, -1])

                multi_task_loss = self.args.rec_weight*rec_loss + self.args.diff_weight*diff_loss + self.args.cl_weight*cl_loss
                

                self.optimizer.zero_grad()
                multi_task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                avg_loss += rec_loss.item()
                
            log_record = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(avg_loss / len(rec_t_data_iter)),
            }

            if (epoch+1) % self.args.print_log_freq == 0:
                print(str(log_record))

            with open(self.args.log_file, "a") as f:
                f.write(str(log_record) + "\n")

        else: # Val or Test
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader)) 
            self.model.eval()
            pred_list, answer_list = None, None


            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids,input_seq_len, target_pos, answers, _, _ = batch
                rec_output = self.seq_model(input_ids)
                rec_output = rec_output[:,-1,:]


                rating_pred = self.predict_full(rec_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()

                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(rating_pred, -20)[:, -20:] 
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind] 
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1] 
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()

                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0) 

            metrices_list, log_info = self.get_full_sort_score(epoch, answer_list, pred_list)


            return metrices_list, log_info
                    
    
    def calculate_diff_loss(self, model, x_start, s, t, noise=None, loss_type="l2"):

        if noise is None:
            noise = torch.randn_like(x_start) 
        
        x_noisy = model.forward_process(x_start=x_start, t=t, noise=noise)

        predicted_x = model(x_noisy, s, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x 
    
    def calculate_cl_loss(self, x0_hat_rep, x0_rep, target):
        batch_size = x0_rep.shape[0]
        sem_nce_logits, sem_nce_labels = self.info_nce(x0_hat_rep,x0_rep,self.args.temperature, batch_size, self.sim, target)
        cl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cl_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot',intent_id=None):

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(intent_id)
        negative_samples = sim
        negative_samples[mask==0]=float("-inf")

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    
    def mask_correlated_samples(self, label):
        label=label.view(1,-1)
        label=label.expand((2,label.shape[-1])).reshape(1,-1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask==0

