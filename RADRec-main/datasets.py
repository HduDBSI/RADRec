import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from utils import generate_rating_matrix_valid, generate_rating_matrix_test


ENTROPY_GROUP_LOW = 0
ENTROPY_GROUP_MID = 1
ENTROPY_GROUP_HIGH = 2

DEFAULT_ENTROPY_THETA = {
    "Beauty": 0.9,
    "Sports_and_Outdoors": 0.9,
    "Toys_and_Games": 0.9,
    "Video": 0.9,
    "ml-1m": 0.9,
}


def resolve_entropy_theta(data_name, explicit_theta=None):
    if explicit_theta is not None:
        return explicit_theta
    return DEFAULT_ENTROPY_THETA.get(data_name, 0.6)


def resolve_entropy_pretrained_path(args):
    candidates = []
    manual_path = getattr(args, "entropy_pretrained_path", None)
    if manual_path:
        candidates.append(manual_path)

    pretrained_dir = getattr(args, "pretrained_dir", "pretrained")
    candidates.append(
        os.path.join(pretrained_dir, f"{args.model_name}-{args.data_name}-{args.model_idx}.pt")
    )
    fallback_path = os.path.join(pretrained_dir, f"{args.model_name}-{args.data_name}-1.pt")
    if fallback_path not in candidates:
        candidates.append(fallback_path)

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def _format_entropy_cache_value(value):
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return str(value).replace(".", "p")


def resolve_entropy_cache_path(args, checkpoint_path, theta, num_sequences):
    segmented_file = getattr(args, "segmented_file", None)
    segmented_stat = os.stat(segmented_file) if segmented_file and os.path.exists(segmented_file) else None
    checkpoint_stat = os.stat(checkpoint_path)

    signature_payload = {
        "data_name": args.data_name,
        "max_seq_length": args.max_seq_length,
        "num_sequences": num_sequences,
        "theta": float(theta),
        "low_entropy_ratio": float(args.low_entropy_ratio),
        "high_entropy_ratio": float(args.high_entropy_ratio),
        "low_coupling_ratio": float(getattr(args, "low_coupling_ratio", args.low_entropy_ratio)),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "segmented_file": os.path.abspath(segmented_file) if segmented_file else None,
        "segmented_size": int(segmented_stat.st_size) if segmented_stat else None,
        "segmented_mtime_ns": int(segmented_stat.st_mtime_ns) if segmented_stat else None,
    }
    signature = hashlib.md5(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    file_name = (
        f"{args.data_name}_len{args.max_seq_length}_n{num_sequences}_"
        f"theta{_format_entropy_cache_value(float(theta))}_"
        f"low{_format_entropy_cache_value(float(args.low_entropy_ratio))}_"
        f"high{_format_entropy_cache_value(float(args.high_entropy_ratio))}_"
        f"ce{_format_entropy_cache_value(float(getattr(args, 'low_coupling_ratio', args.low_entropy_ratio)))}_"
        f"{checkpoint_tag}_{signature}.npz"
    )
    return os.path.join(args.output_dir, "entropy_cache", file_name)


def D(i_file,o_file,max_len):

    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    max_save_len=max_len+3
    max_keep_len=max_len+2
    for d_ in data:
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        item[-1]=str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start=0
        j=3
        if len(item)>max_save_len:
            while start<len(item)-max_keep_len:
                j=start+4
                while j<len(item):
                    if start<1 and j-start<max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j+=1
                    else:
                        aug_d[u_i].append(item[start:start+max_save_len])
                        break
                start+=1
        else:
            while j<len(item):
                aug_d[u_i].append(item[start:j+1])
                j+=1
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")

def D_random(i_file, o_file, max_len):

    with open(i_file, "r+") as fr:
        data = fr.readlines()
    
    aug_d = {}
    max_save_len = max_len + 3

    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        
        start = 0
        j = 3
        if len(item) > max_save_len:
            while start < len(item) - (max_len + 2):
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        random_sample = random.sample(item[start:j], len(item[start:j]))
                        aug_d[u_i].append(random_sample)
                        j += 1
                    else:
                        random_sample = random.sample(item[start:start + max_save_len], len(item[start:start + max_save_len]))
                        aug_d[u_i].append(random_sample)
                        break
                start += 1
        else:
            while j < len(item):
                random_sample = random.sample(item[start:j + 1], len(item[start:j + 1]))
                aug_d[u_i].append(random_sample)
                j += 1

    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")

def get_seqs_and_matrixes(type, data_file):
    user_seq = []
    user_id = []
    item_set = set()
    
    with open(data_file, "r") as fr:
        for line in fr:
            parts = line.strip().split()
            user = int(parts[0])
            items = list(map(int, parts[1:]))
            user_id.append(user)
            user_seq.append(items)
            item_set.update(items)
    
    max_item = max(item_set)
    num_users = len(user_id)
    num_items = max_item + 2
    
    if type == "training":
        return user_id, user_seq
    elif type == "rating":
        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
        return user_id, user_seq, max_item, valid_rating_matrix, test_rating_matrix
    else:
        raise NotImplementedError


class DatasetForRADRec(Dataset):
    _embedding_cache = {}
    _entropy_cache = {}

    def __init__(self, args, user_seq, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.entropy_scores = np.zeros(len(self.user_seq), dtype=np.float32)
        self.entropy_groups = np.full(len(self.user_seq), ENTROPY_GROUP_MID, dtype=np.int64)
        self.entropy_metadata = {}

        if self.data_type == "train":
            self.entropy_scores, self.entropy_groups, self.entropy_metadata = self._get_entropy_groups()


    def __getitem__(self, index):
        user_id = index
        item_seq = self.user_seq[user_id]

        if self.data_type == "train":
            input_seq = item_seq[ : -3]
            target_pos = item_seq[1 : -2]
            answer = item_seq[-3]

        elif self.data_type == "valid":
            input_seq = item_seq[:-2]
            target_pos = item_seq[1:-1]
            answer = [item_seq[-2]]
        
        else:
            item_seq_with_noise = self._add_noise_for_robustness(item_seq)
            input_seq = item_seq_with_noise[:-1]
            target_pos = item_seq_with_noise[1:]
            answer = [item_seq_with_noise[-1]]

        entropy_score = self.entropy_scores[index]
        entropy_group = self.entropy_groups[index]
        cur_rec_tensors = self._data_construction(
            user_id,
            input_seq,
            target_pos,
            answer,
            entropy_score,
            entropy_group,
        )

        return cur_rec_tensors


    def _data_construction(self, user_id, input_seq, target_pos, answer, entropy_score, entropy_group):
        copied_input_seq = list(input_seq)
        input_seq_len = len(copied_input_seq)
        pad_len = self.max_len - input_seq_len
        copied_input_seq =[0] * pad_len+copied_input_seq
        copied_input_seq=copied_input_seq[-self.max_len:]

        target_pos =  [0] * pad_len+target_pos
        target_pos = target_pos[-self.max_len:]

        assert len(target_pos) == self.max_len
        assert len(copied_input_seq) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(copied_input_seq, dtype=torch.long),
            torch.tensor(input_seq_len, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(entropy_score, dtype=torch.float),
            torch.tensor(entropy_group, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_for_robustness(self, items):
        if self.args.noise_ratio == 0:
            return items
        copied_sequence = list(items)
        insert_nums = int(self.args.noise_ratio * len(copied_sequence))

        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __len__(self):

        return len(self.user_seq)

    def _get_entropy_groups(self):
        checkpoint_path = resolve_entropy_pretrained_path(self.args)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Cannot find entropy pretrained checkpoint at '{checkpoint_path}'. "
                "Pass --entropy_pretrained_path explicitly if you want to use a different file."
            )

        theta = resolve_entropy_theta(self.args.data_name, getattr(self.args, "entropy_theta", None))
        cache_path = resolve_entropy_cache_path(
            self.args,
            checkpoint_path=checkpoint_path,
            theta=theta,
            num_sequences=len(self.user_seq),
        )
        allow_entropy_cache_build = getattr(self.args, "allow_entropy_cache_build", False)
        rebuild_entropy_cache = getattr(self.args, "rebuild_entropy_cache", False)
        cache_key = (
            os.path.abspath(checkpoint_path),
            self.args.data_name,
            self.max_len,
            len(self.user_seq),
            float(theta),
            float(self.args.low_entropy_ratio),
            float(self.args.high_entropy_ratio),
            float(getattr(self.args, "low_coupling_ratio", self.args.low_entropy_ratio)),
        )

        if rebuild_entropy_cache and cache_key in self._entropy_cache:
            del self._entropy_cache[cache_key]

        if cache_key not in self._entropy_cache:
            cached_entropy = None if rebuild_entropy_cache else self._load_entropy_cache_from_file(cache_path)
            if cached_entropy is not None:
                self._entropy_cache[cache_key] = cached_entropy
                _, _, metadata = cached_entropy
                print(
                    "Loaded entropy split cache | "
                    f"dataset={self.args.data_name} | theta={theta} | "
                    f"low/mid/high={metadata['low_count']}/{metadata['mid_count']}/{metadata['high_count']} | "
                    f"cache={cache_path}"
                )
            else:
                if not allow_entropy_cache_build:
                    raise FileNotFoundError(
                        f"Entropy cache not found or built with an incompatible partition strategy at '{cache_path}'. "
                        "Build it first with `python3 main.py --build_entropy_cache_only ...`."
                    )
                pretrained_item_embeddings = self._load_pretrained_item_embeddings(checkpoint_path)
                entropy_scores, coupling_scores = self._compute_interest_structure_scores(
                    pretrained_item_embeddings=pretrained_item_embeddings,
                    user_sequences=self.user_seq,
                    max_len=self.max_len,
                    batch_size=max(1, getattr(self.args, "entropy_batch_size", self.args.batch_size)),
                    theta=theta,
                )
                entropy_groups, metadata = self._assign_entropy_groups(
                    entropy_scores,
                    low_ratio=self.args.low_entropy_ratio,
                    high_ratio=self.args.high_entropy_ratio,
                    coupling_scores=coupling_scores,
                    low_coupling_ratio=getattr(self.args, "low_coupling_ratio", self.args.low_entropy_ratio),
                )
                metadata.update(
                    {
                        "theta": theta,
                        "checkpoint_path": checkpoint_path,
                        "partition_strategy": "se_ce_then_entropy",
                    }
                )
                self._save_entropy_cache_to_file(
                    cache_path,
                    entropy_scores=entropy_scores,
                    entropy_groups=entropy_groups,
                    metadata=metadata,
                )
                metadata["cache_path"] = cache_path
                self._entropy_cache[cache_key] = (entropy_scores, entropy_groups, metadata)
                print(
                    "Entropy split prepared | "
                    f"dataset={self.args.data_name} | theta={theta} | "
                    f"low/mid/high={metadata['low_count']}/{metadata['mid_count']}/{metadata['high_count']} | "
                    f"cache={cache_path}"
                )

        return self._entropy_cache[cache_key]

    @classmethod
    def _load_pretrained_item_embeddings(cls, checkpoint_path):
        cache_key = os.path.abspath(checkpoint_path)
        if cache_key in cls._embedding_cache:
            return cls._embedding_cache[cache_key]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format in '{checkpoint_path}'.")

        for key in ("item_embeddings.weight", "seq_model.item_embeddings.weight"):
            if key in state_dict:
                cls._embedding_cache[cache_key] = state_dict[key].detach().cpu()
                return cls._embedding_cache[cache_key]

        raise KeyError(
            f"Cannot find item embedding weights in '{checkpoint_path}'. "
            "Expected 'item_embeddings.weight' or 'seq_model.item_embeddings.weight'."
        )

    @classmethod
    def _compute_interest_structure_scores(
        cls,
        pretrained_item_embeddings,
        user_sequences,
        max_len,
        batch_size,
        theta,
    ):
        entropy_scores = np.zeros(len(user_sequences), dtype=np.float32)
        coupling_scores = np.zeros(len(user_sequences), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(user_sequences), batch_size):
                end = min(start + batch_size, len(user_sequences))
                batch_sequences = []
                batch_lengths = []

                for item_seq in user_sequences[start:end]:
                    input_seq = item_seq[:-3]
                    input_seq = input_seq[-max_len:]
                    batch_lengths.append(len(input_seq))
                    padded_input_seq = [0] * (max_len - len(input_seq)) + input_seq
                    batch_sequences.append(padded_input_seq)

                item_seq_tensor = torch.tensor(batch_sequences, dtype=torch.long)
                item_seq_len_tensor = torch.tensor(batch_lengths, dtype=torch.long)
                embeddings = pretrained_item_embeddings[item_seq_tensor]
                batch_entropy, batch_coupling = cls._interest_structure_from_embeddings(
                    embeddings=embeddings,
                    item_seq_len=item_seq_len_tensor,
                    theta=theta,
                )
                entropy_scores[start:end] = batch_entropy.cpu().numpy()
                coupling_scores[start:end] = batch_coupling.cpu().numpy()

        return entropy_scores, coupling_scores

    @classmethod
    def _interest_structure_from_embeddings(cls, embeddings, item_seq_len, theta):
        batch_size = embeddings.shape[0]
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
        adj_matrix = (sim_matrix > theta).detach().cpu().numpy()

        ie_values = []
        ce_values = []
        for batch_index in range(batch_size):
            seq_len = int(item_seq_len[batch_index].item())
            if seq_len <= 1:
                ie_values.append(torch.tensor(0.0, dtype=embeddings.dtype))
                ce_values.append(torch.tensor(0.0, dtype=embeddings.dtype))
                continue

            valid_adj = adj_matrix[batch_index, -seq_len:, -seq_len:]
            valid_embeddings = embeddings[batch_index, -seq_len:, :]
            components = cls._get_connected_components(valid_adj)
            component_sizes = [len(component) for component in components]
            class_probs = torch.tensor(
                [size / seq_len for size in component_sizes],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            entropy = -torch.sum(class_probs * torch.log2(class_probs + 1e-10))
            coupling = cls._coupling_from_components(
                embeddings=valid_embeddings,
                components=components,
                class_probs=class_probs,
            )
            ie_values.append(entropy)
            ce_values.append(coupling)

        return torch.stack(ie_values), torch.stack(ce_values)

    @staticmethod
    def _coupling_from_components(embeddings, components, class_probs):
        if len(components) <= 1:
            return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device)

        centroids = []
        for component in components:
            component_indices = torch.tensor(component, dtype=torch.long, device=embeddings.device)
            centroids.append(embeddings.index_select(0, component_indices).mean(dim=0))
        centroids = torch.stack(centroids, dim=0)
        centroids = F.normalize(centroids, p=2, dim=-1)
        sim_matrix = torch.matmul(centroids, centroids.transpose(0, 1))
        off_diagonal_mask = ~torch.eye(
            len(components),
            dtype=torch.bool,
            device=embeddings.device,
        )
        pairwise_mean = sim_matrix.masked_select(off_diagonal_mask).view(len(components), -1).mean(dim=1)
        return torch.sum(class_probs.to(embeddings.device) * pairwise_mean)

    @staticmethod
    def _get_connected_components(adj_matrix):
        seq_len = adj_matrix.shape[0]
        if seq_len == 0:
            return [[0]]

        visited = np.zeros(seq_len, dtype=bool)
        components = []

        for node in range(seq_len):
            if visited[node]:
                continue

            stack = [node]
            visited[node] = True
            component = []

            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = np.flatnonzero(adj_matrix[current])
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(int(neighbor))

            components.append(component)

        return components or [[0]]

    @staticmethod
    def _assign_entropy_groups(
        entropy_scores,
        low_ratio,
        high_ratio,
        coupling_scores=None,
        low_coupling_ratio=None,
    ):
        num_samples = len(entropy_scores)
        groups = np.full(num_samples, ENTROPY_GROUP_MID, dtype=np.int64)
        robust_mask = np.ones(num_samples, dtype=bool)

        low_boundary = None
        high_boundary = None
        low_coupling_threshold = None
        vulnerable_count = 0

        if num_samples > 0 and coupling_scores is not None and low_coupling_ratio is not None and low_coupling_ratio > 0:
            low_boundary = float(np.quantile(entropy_scores, low_ratio))
            high_boundary = float(np.quantile(entropy_scores, 1 - high_ratio))
            low_coupling_threshold = float(np.quantile(coupling_scores, low_coupling_ratio))
            vulnerable_mask = (
                (entropy_scores >= low_boundary)
                & (entropy_scores <= high_boundary)
                & (coupling_scores <= low_coupling_threshold)
            )
            robust_mask[vulnerable_mask] = False
            vulnerable_count = int(vulnerable_mask.sum())

        robust_indices = np.flatnonzero(robust_mask)
        sorted_indices = robust_indices[np.argsort(entropy_scores[robust_indices], kind="stable")]
        num_robust_samples = len(sorted_indices)

        low_count = int(num_robust_samples * low_ratio)
        high_count = int(num_robust_samples * high_ratio)

        if low_ratio > 0 and low_count == 0 and num_robust_samples >= 3:
            low_count = 1
        if high_ratio > 0 and high_count == 0 and num_robust_samples - low_count >= 2:
            high_count = 1

        high_count = min(high_count, max(0, num_robust_samples - low_count))

        if low_count > 0:
            groups[sorted_indices[:low_count]] = ENTROPY_GROUP_LOW
        if high_count > 0:
            groups[sorted_indices[-high_count:]] = ENTROPY_GROUP_HIGH

        metadata = {
            "low_count": int(low_count),
            "mid_count": int(num_samples - low_count - high_count),
            "high_count": int(high_count),
            "low_threshold": float(entropy_scores[sorted_indices[low_count - 1]]) if low_count > 0 else None,
            "high_threshold": float(entropy_scores[sorted_indices[-high_count]]) if high_count > 0 else None,
            "vulnerable_count": vulnerable_count,
            "low_coupling_ratio": float(low_coupling_ratio) if low_coupling_ratio is not None else None,
            "mid_entropy_low_boundary": low_boundary,
            "mid_entropy_high_boundary": high_boundary,
            "low_coupling_threshold": low_coupling_threshold,
        }
        return groups, metadata

    @staticmethod
    def _save_entropy_cache_to_file(cache_path, entropy_scores, entropy_groups, metadata):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        serializable_metadata = dict(metadata)
        serializable_metadata["cache_path"] = str(cache_path)
        np.savez_compressed(
            cache_path,
            entropy_scores=np.asarray(entropy_scores, dtype=np.float32),
            entropy_groups=np.asarray(entropy_groups, dtype=np.int64),
            metadata_json=np.array(json.dumps(serializable_metadata, ensure_ascii=True)),
        )

    @staticmethod
    def _load_entropy_cache_from_file(cache_path):
        if not os.path.exists(cache_path):
            return None

        with np.load(cache_path, allow_pickle=False) as cached_data:
            required_keys = {"entropy_scores", "entropy_groups", "metadata_json"}
            if not required_keys.issubset(cached_data.files):
                return None
            entropy_scores = cached_data["entropy_scores"].astype(np.float32)
            entropy_groups = cached_data["entropy_groups"].astype(np.int64)
            metadata = json.loads(cached_data["metadata_json"].item())
        if metadata.get("partition_strategy") != "se_ce_then_entropy":
            return None
        metadata["cache_path"] = cache_path
        return entropy_scores, entropy_groups, metadata
