# _*_ coding: utf-8 _*_

import argparse
import copy

from tqdm import tqdm
import itertools
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--dev_run", action="store_true")
parser.add_argument(
    "--working_aspect_idx",
    type=int,
    default=0,
    help="Apect index as in [violence, substance, sex, consumerism, positive].",
)
parser.add_argument("--base_dir", type=str, default="../processed_data/emotion_emb/")
parser.add_argument("--model_save_dir", type=str, default="./save/")
parser.add_argument("--log_save_dir", type=str, default="./record/")

parser.add_argument("--use_gpu_idx", type=int, default=5)

parser.add_argument(
    "--train_batch_size", type=int, default=40, help="train_batch_size."
)
parser.add_argument("--dev_batch_size", type=int, default=2, help="dev_batch_size.")
parser.add_argument("--test_batch_size", type=int, default=1, help="test_batch_size.")

parser.add_argument("--slate_num", type=int, default=2, help="compare num.")
parser.add_argument("--rank_output_size", type=int, default=3, help="rank output num.")
parser.add_argument("--cls_output_size", type=int, default=3, help="class num.")
parser.add_argument("--input_size", type=int, default=768, help="input dimension.")
parser.add_argument(
    "--hidden_size", type=int, default=200, help="RNN hidden dimension."
)
parser.add_argument(
    "--projection_size", type=int, default=50, help="projection_size dimension."
)

parser.add_argument("--lr", type=float, default=1e-3, help="learning rate.")
parser.add_argument("--training_epochs", type=int, default=200, help="Training epochs.")
parser.add_argument("--patience", type=int, default=30, help="Early stop patience.")
parser.add_argument(
    "--multiple_runs", type=int, default=5, help="Multiple runs of experiment."
)
parser.add_argument("--numpy_seed", type=int, default=42, help="NumPy seed.")

args = parser.parse_args()

# args = parser.parse_args(args=['--dev_run', '--training_epochs', '5'])
# args = parser.parse_args(args=['--training_epochs', '5'])


doc_list = ["violence", "substance", "sex", "consumerism", "positive"]

# working_aspect = doc_list[args.working_aspect_idx]
working_aspect = "all5ava"


print("Now working on:", working_aspect)
text_col = 3
emo_col = 4


big_df = pd.read_pickle(args.base_dir + "/lyrics_emb.pkl")


def get_train_test_idx(lst, i):
    n = int(len(lst) / 10)
    start = i * n
    end = (i + 1) * n
    train_idx = lst[0:start] + lst[end:]

    test_idx = lst[start:end]
    return train_idx, test_idx


class ExperimentLog:
    def __init__(self, logfile):
        self.logfilename = logfile

    def __call__(self, content):
        with open(self.logfilename, "a") as f_log:
            f_log.write(content)
            f_log.write("\n")


logger = ExperimentLog(args.log_save_dir + "record_cpr_10CV.txt")

### 5 rand seed
random_seeds = [0, 1, 42, 2021, 2022]
all_seed_f1_collection = []

for each_seed in random_seeds:
    logger("Random seed: " + str(each_seed))

    np.random.seed(each_seed)
    item_idx_list = np.arange(len(big_df))
    np.random.shuffle(item_idx_list)
    shuffled_list = list(item_idx_list)

    folds_f1_collection = []

    for counting in range(10):
        print(counting, "-th fold.--------")
        logger(str(counting) + "-th fold.--------")

        train_idx, test_idx = get_train_test_idx(shuffled_list, counting)

        train_all = big_df.loc[train_idx]
        test_raw_data = big_df.loc[test_idx]

        train_raw_data, dev_raw_data = train_test_split(
            train_all, test_size=1.0 / 9.0, random_state=each_seed
        )

        to_device = "cuda:" + str(args.use_gpu_idx)

        def get_target_column(matrix, i):
            return torch.stack([row[i] for row in matrix])

        def get_column(matrix, i):
            return [row[i] for row in matrix]

        def compare_pair(lst):
            result = 0
            # compare left to right, if left < right return 0...
            if lst[0] < lst[1]:
                result = 0
            elif lst[0] == lst[1]:
                result = 1
            else:
                result = 2
            return result

        def label_to_pair_compare(lst):
            return torch.Tensor([compare_pair(each) for each in lst]).long()

        def reshape_cls_output(list_for_class):
            final_cls_out = torch.stack(list_for_class)  # [[1,3,5],[2,4,6]]
            final_cls_out = final_cls_out.permute(1, 0, 2)
            final_cls_out = final_cls_out.reshape(
                -1, args.cls_output_size
            )  # [num_samples * num_class]
            return final_cls_out

        class SiameseLSTM(nn.Module):
            def __init__(
                self,
                input_size,
                slate_num,
                rank_output_size,
                class_num,
                hidden_size,
                projection_size,
            ):
                super(SiameseLSTM, self).__init__()

                bsz = 1
                self.direction = 2
                self.input_size = input_size
                self.slate_num = slate_num
                self.rank_output_size = rank_output_size
                self.class_num = class_num
                self.hidden_size = hidden_size
                self.projection_size = projection_size * 2
                self.batch_size = bsz

                self.lstm_sem = nn.LSTM(
                    self.input_size,
                    self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )

                self.lstm_emo = nn.LSTM(
                    self.input_size,
                    self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )

                self.nonlinear = nn.LeakyReLU()

                self.vio_proj = nn.Linear(
                    self.hidden_size * self.direction * 2, self.projection_size
                )
                self.sub_proj = nn.Linear(
                    self.hidden_size * self.direction * 2, self.projection_size
                )
                self.sex_proj = nn.Linear(
                    self.hidden_size * self.direction * 2, self.projection_size
                )
                self.con_proj = nn.Linear(
                    self.hidden_size * self.direction * 2, self.projection_size
                )
                self.pos_proj = nn.Linear(
                    self.hidden_size * self.direction * 2, self.projection_size
                )

                self.vio_matrix = nn.Parameter(
                    torch.randn(self.projection_size, self.projection_size)
                )
                self.sub_matrix = nn.Parameter(
                    torch.randn(self.projection_size, self.projection_size)
                )
                self.sex_matrix = nn.Parameter(
                    torch.randn(self.projection_size, self.projection_size)
                )
                self.con_matrix = nn.Parameter(
                    torch.randn(self.projection_size, self.projection_size)
                )
                self.pos_matrix = nn.Parameter(
                    torch.randn(self.projection_size, self.projection_size)
                )

                self.vio_classifier = nn.Linear(self.projection_size, self.class_num)
                self.sub_classifier = nn.Linear(self.projection_size, self.class_num)
                self.sex_classifier = nn.Linear(self.projection_size, self.class_num)
                self.con_classifier = nn.Linear(self.projection_size, self.class_num)
                self.pos_classifier = nn.Linear(self.projection_size, self.class_num)

                self.vio_ranker = nn.Linear(
                    self.projection_size * 2, self.rank_output_size
                )
                self.sub_ranker = nn.Linear(
                    self.projection_size * 2, self.rank_output_size
                )
                self.sex_ranker = nn.Linear(
                    self.projection_size * 2, self.rank_output_size
                )
                self.con_ranker = nn.Linear(
                    self.projection_size * 2, self.rank_output_size
                )
                self.pos_ranker = nn.Linear(
                    self.projection_size * 2, self.rank_output_size
                )

            def forward_one(self, x, batch_size=None):
                sent_sem = [each[0] for each in x]
                sent_emo = [each[1] for each in x]
                lens = [len(sq) for sq in sent_sem]

                sent_sem = pad_sequence(sent_sem, batch_first=True, padding_value=0).to(
                    device=to_device
                )
                sent_sem = pack_padded_sequence(
                    sent_sem, lens, batch_first=True, enforce_sorted=False
                )

                sent_emo = pad_sequence(sent_emo, batch_first=True, padding_value=0).to(
                    device=to_device
                )
                sent_emo = pack_padded_sequence(
                    sent_emo, lens, batch_first=True, enforce_sorted=False
                )

                if batch_size is None:
                    # Initial hidden state of the LSTM (num_layers * num_directions, batch, hidden_size)
                    h_0 = (
                        torch.zeros(
                            1 * self.direction, self.batch_size, self.hidden_size
                        )
                        .requires_grad_()
                        .to(device=to_device)
                    )
                    # Initial cell state of the LSTM
                    c_0 = (
                        torch.zeros(
                            1 * self.direction, self.batch_size, self.hidden_size
                        )
                        .requires_grad_()
                        .to(device=to_device)
                    )
                    h_1 = (
                        torch.zeros(
                            1 * self.direction, self.batch_size, self.hidden_size
                        )
                        .requires_grad_()
                        .to(device=to_device)
                    )
                    # Initial cell state of the LSTM
                    c_1 = (
                        torch.zeros(
                            1 * self.direction, self.batch_size, self.hidden_size
                        )
                        .requires_grad_()
                        .to(device=to_device)
                    )

                else:
                    h_0 = (
                        torch.zeros(1 * self.direction, batch_size, self.hidden_size)
                        .requires_grad_()
                        .to(device=to_device)
                    )
                    c_0 = (
                        torch.zeros(1 * self.direction, batch_size, self.hidden_size)
                        .requires_grad_()
                        .to(device=to_device)
                    )
                    h_1 = (
                        torch.zeros(1 * self.direction, batch_size, self.hidden_size)
                        .requires_grad_()
                        .to(device=to_device)
                    )
                    c_1 = (
                        torch.zeros(1 * self.direction, batch_size, self.hidden_size)
                        .requires_grad_()
                        .to(device=to_device)
                    )

                output_sem, (final_hidden_state, final_cell_state) = self.lstm_sem(
                    sent_sem, (h_0, c_0)
                )
                output_emo, (final_hidden_state, final_cell_state) = self.lstm_emo(
                    sent_emo, (h_1, c_1)
                )

                output_sem = pad_packed_sequence(
                    output_sem, batch_first=True
                )  # padded seq, lengths
                output_emo = pad_packed_sequence(
                    output_emo, batch_first=True
                )  # padded seq, lengths
                output_sem = torch.max(output_sem[0], dim=1)[
                    0
                ]  # after max, (max tensor, max_indices)
                output_emo = torch.max(output_emo[0], dim=1)[
                    0
                ]  # after max, (max tensor, max_indices)
                output = torch.cat((output_sem, output_emo), 1)

                vio_proj_output = self.nonlinear(self.vio_proj(output))
                sub_proj_output = self.nonlinear(self.sub_proj(output))
                sex_proj_output = self.nonlinear(self.sex_proj(output))
                con_proj_output = self.nonlinear(self.con_proj(output))
                pos_proj_output = self.nonlinear(self.pos_proj(output))

                vio_att_applied = torch.add(
                    vio_proj_output,
                    torch.mul(
                        vio_proj_output,
                        F.softmax(
                            torch.matmul(vio_proj_output, self.vio_matrix), dim=-1
                        ),
                    ),
                )
                sub_att_applied = torch.add(
                    sub_proj_output,
                    torch.mul(
                        sub_proj_output,
                        F.softmax(
                            torch.matmul(sub_proj_output, self.sub_matrix), dim=-1
                        ),
                    ),
                )
                sex_att_applied = torch.add(
                    sex_proj_output,
                    torch.mul(
                        sex_proj_output,
                        F.softmax(
                            torch.matmul(sex_proj_output, self.sex_matrix), dim=-1
                        ),
                    ),
                )
                con_att_applied = torch.add(
                    con_proj_output,
                    torch.mul(
                        con_proj_output,
                        F.softmax(
                            torch.matmul(con_proj_output, self.con_matrix), dim=-1
                        ),
                    ),
                )
                pos_att_applied = torch.add(
                    pos_proj_output,
                    torch.mul(
                        pos_proj_output,
                        F.softmax(
                            torch.matmul(pos_proj_output, self.pos_matrix), dim=-1
                        ),
                    ),
                )

                vio_output = self.vio_classifier(vio_att_applied)
                sub_output = self.sub_classifier(sub_att_applied)
                sex_output = self.sex_classifier(sex_att_applied)
                con_output = self.con_classifier(con_att_applied)
                pos_output = self.pos_classifier(pos_att_applied)

                final_cls_output = (
                    vio_output,
                    sub_output,
                    sex_output,
                    con_output,
                    pos_output,
                )
                all_att_representation = (
                    vio_att_applied,
                    sub_att_applied,
                    sex_att_applied,
                    con_att_applied,
                    pos_att_applied,
                )

                return final_cls_output, all_att_representation

            def forward(self, x, batch_size=None):
                current_bsz = len(x)

                # vio, sub, sex, con, pos lists
                list_for_class = [[], [], [], [], []]
                list_for_rank = [[], [], [], [], []]
                # x shape: batch * ranklists
                # [[1,2],
                # [3,4],
                # [5,6],
                # ...
                # [99,100]]

                for i in range(len(x[0])):  # 2 dim -> # [[1,3,5],[2,4,6]]
                    # one column is one batch, feed in one batch of
                    cls_output, rank_output = self.forward_one(
                        get_column(x, i), batch_size=current_bsz
                    )
                    # todo: unpack rank representation, reshape cls result
                    for i in range(len(cls_output)):
                        list_for_class[i].append(cls_output[i])  # one rank output
                        list_for_rank[i].append(
                            rank_output[i]
                        )  # one column classification output = batch size * num_class
                # 2 dim -> # [[1,3,5],[2,4,6]]
                championship_vio = torch.cat(
                    list_for_rank[0], dim=1
                )  # [[1,2],[3,4],[5,6]]
                championship_sub = torch.cat(
                    list_for_rank[1], dim=1
                )  # [[1,2],[3,4],[5,6]]
                championship_sex = torch.cat(
                    list_for_rank[2], dim=1
                )  # [[1,2],[3,4],[5,6]]
                championship_con = torch.cat(
                    list_for_rank[3], dim=1
                )  # [[1,2],[3,4],[5,6]]
                championship_pos = torch.cat(
                    list_for_rank[4], dim=1
                )  # [[1,2],[3,4],[5,6]]

                final_rank_out_vio = self.vio_ranker(championship_vio)
                final_rank_out_sub = self.sub_ranker(championship_sub)
                final_rank_out_sex = self.sex_ranker(championship_sex)
                final_rank_out_con = self.con_ranker(championship_con)
                final_rank_out_pos = self.pos_ranker(championship_pos)

                final_rank_out = (
                    final_rank_out_vio,
                    final_rank_out_sub,
                    final_rank_out_sex,
                    final_rank_out_con,
                    final_rank_out_pos,
                )

                ### double check, unpack cls result
                final_cls_out_vio = reshape_cls_output(list_for_class[0])
                final_cls_out_sub = reshape_cls_output(list_for_class[1])
                final_cls_out_sex = reshape_cls_output(list_for_class[2])
                final_cls_out_con = reshape_cls_output(list_for_class[3])
                final_cls_out_pos = reshape_cls_output(list_for_class[4])

                final_cls_out = (
                    final_cls_out_vio,
                    final_cls_out_sub,
                    final_cls_out_sex,
                    final_cls_out_con,
                    final_cls_out_pos,
                )

                return final_rank_out, final_cls_out

        def loss_function(prediction, target):
            return F.cross_entropy(prediction, target)

        class EarlyStopping:
            def __init__(self, patience=10, verbose=False, delta=0):
                self.monitor_criteria = "F1"
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.positive_criteria = True
                self.val_criteria_min = -1
                self.delta = delta
                self.new_best = False

            def __call__(self, val_criteria, model, path):
                print("val_criteria={}".format(val_criteria))
                score = val_criteria
                # initialize
                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_criteria, model, path)
                    self.new_best = True
                # if new run worse than previous best
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    self.new_best = False
                    if self.verbose:
                        print(
                            f"No improvement. EarlyStopping counter: {self.counter} out of {self.patience}"
                        )
                    if self.counter >= self.patience:
                        self.early_stop = True
                # see new best, save model, reset counter
                else:
                    self.best_score = score
                    self.save_checkpoint(val_criteria, model, path)
                    self.counter = 0
                    self.new_best = True

            def save_checkpoint(self, val_criteria, model, path):
                if self.verbose:
                    print(
                        f"Validation F1 improved ({self.val_criteria_min:.6f} --> {val_criteria:.6f}).  Saving model ..."
                    )
                torch.save(
                    model.state_dict(),
                    path
                    + "/"
                    + "model_checkpoint_seed_"
                    + working_aspect
                    + str(each_seed)
                    + ".pth",
                )
                self.val_criteria_min = val_criteria

        class LyricsDataset(torch.utils.data.Dataset):
            def __init__(self, tabular):
                if isinstance(tabular, str):
                    self.annotations = pd.read_csv(tabular, sep="\t")
                else:
                    self.annotations = tabular

            def __len__(self):
                return len(self.annotations)

            def __getitem__(self, index):
                text = self.annotations.iloc[index, text_col]  # -2 is sent emb index
                emo = self.annotations.iloc[index, emo_col]  #
                vio_label = torch.tensor(
                    int(self.annotations.iloc[index, -5])
                )  # -5 to -1
                sub_label = torch.tensor(int(self.annotations.iloc[index, -4]))
                sex_label = torch.tensor(int(self.annotations.iloc[index, -3]))
                con_label = torch.tensor(int(self.annotations.iloc[index, -2]))
                pos_label = torch.tensor(int(self.annotations.iloc[index, -1]))
                y_label = torch.stack(
                    [vio_label, sub_label, sex_label, con_label, pos_label]
                )

                return {"text": (text, emo), "label": y_label}

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        def my_collate_fn(batch):
            # batch size * 2 samples
            text_batch = [each_item["text"] for each_item in batch]

            # reshape to [batch size * 2]
            text_batch = list(chunks(text_batch, args.slate_num))

            label_batch = torch.stack(
                [each_item["label"] for each_item in batch]
            ).long()
            # reshape to [batch size * 2]

            return text_batch, label_batch

        def test_collate_fn(batch):
            text_batch = [batch[0]["text"], batch[0]["text"]]

            # reshape to [batch size * 2]
            text_batch = list(chunks(text_batch, args.slate_num))

            label_batch = torch.stack([batch[0]["label"], batch[0]["label"]]).long()

            return text_batch, label_batch

        model = SiameseLSTM(
            args.input_size,
            args.slate_num,
            args.rank_output_size,
            args.cls_output_size,
            args.hidden_size,
            args.projection_size,
        ).to(to_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_loss = []
        valid_loss = []
        train_all_epochs_loss = []
        valid_epochs_loss = []

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        train_dataset = LyricsDataset(train_raw_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=my_collate_fn,
            num_workers=10,
            drop_last=True,
        )

        dev_dataset = LyricsDataset(dev_raw_data)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.dev_batch_size,
            shuffle=False,
            collate_fn=my_collate_fn,
            drop_last=True,
        )

        test_dataset = LyricsDataset(test_raw_data)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            collate_fn=test_collate_fn,
            drop_last=True,
        )

        if args.dev_run:
            args.training_epochs = 1

        for epoch in range(args.training_epochs):
            print("====== {}-th epoch ======".format(epoch))
            logger("====== {}-th epoch ======".format(epoch))

            # if not dev run, start training
            if args.dev_run != True:
                model.train()
                train_epoch_loss = []
                # enumerate(tqdm(x))
                # text, target = batch
                with tqdm(train_loader, unit=" batch") as tepoch:
                    for idx, (text, target) in enumerate(tepoch):
                        #                     target = target.to(to_device)
                        vio_target = get_target_column(target, 0).to(to_device)
                        sub_target = get_target_column(target, 1).to(to_device)
                        sex_target = get_target_column(target, 2).to(to_device)
                        con_target = get_target_column(target, 3).to(to_device)
                        pos_target = get_target_column(target, 4).to(to_device)

                        vio_pairwise_target = label_to_pair_compare(
                            list(chunks(vio_target, args.slate_num))
                        ).to(vio_target.device)
                        sub_pairwise_target = label_to_pair_compare(
                            list(chunks(sub_target, args.slate_num))
                        ).to(sub_target.device)
                        sex_pairwise_target = label_to_pair_compare(
                            list(chunks(sex_target, args.slate_num))
                        ).to(sex_target.device)
                        con_pairwise_target = label_to_pair_compare(
                            list(chunks(con_target, args.slate_num))
                        ).to(con_target.device)
                        pos_pairwise_target = label_to_pair_compare(
                            list(chunks(pos_target, args.slate_num))
                        ).to(pos_target.device)

                        #                     prediction = model(text, batch_size = len(target))
                        rank_pred, cls_pred = model(text, batch_size=len(target))

                        vio_cls_loss = loss_function(cls_pred[0], vio_target)
                        sub_cls_loss = loss_function(cls_pred[1], sub_target)
                        sex_cls_loss = loss_function(cls_pred[2], sex_target)
                        con_cls_loss = loss_function(cls_pred[3], con_target)
                        pos_cls_loss = loss_function(cls_pred[4], pos_target)

                        vio_rank_loss = loss_function(rank_pred[0], vio_pairwise_target)
                        sub_rank_loss = loss_function(rank_pred[1], sub_pairwise_target)
                        sex_rank_loss = loss_function(rank_pred[2], sex_pairwise_target)
                        con_rank_loss = loss_function(rank_pred[3], con_pairwise_target)
                        pos_rank_loss = loss_function(rank_pred[4], pos_pairwise_target)

                        loss = (
                            vio_cls_loss
                            + sub_cls_loss
                            + sex_cls_loss
                            + con_cls_loss
                            + pos_cls_loss
                            + vio_rank_loss
                            + sub_rank_loss
                            + sex_rank_loss
                            + con_rank_loss
                            + pos_rank_loss
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        tepoch.set_postfix(loss=loss.item())
                        train_epoch_loss.append(loss.item())
                #             train_loss.append(loss.item()) # track step loss

                print(
                    "epoch={}/{}, epoch loss={}".format(
                        epoch, args.training_epochs, np.average(train_epoch_loss)
                    )
                )
                train_all_epochs_loss.append(
                    np.average(train_epoch_loss)
                )  # track epoch loss

            # =====================validation============================
            model.eval()
            val_step_outputs = []
            with tqdm(dev_loader, unit=" batch") as vepoch:
                for idx, (text, target) in enumerate(vepoch):
                    vio_target = get_target_column(target, 0).to(to_device)
                    sub_target = get_target_column(target, 1).to(to_device)
                    sex_target = get_target_column(target, 2).to(to_device)
                    con_target = get_target_column(target, 3).to(to_device)
                    pos_target = get_target_column(target, 4).to(to_device)

                    vio_pairwise_target = label_to_pair_compare(
                        list(chunks(vio_target, args.slate_num))
                    ).to(vio_target.device)
                    sub_pairwise_target = label_to_pair_compare(
                        list(chunks(sub_target, args.slate_num))
                    ).to(sub_target.device)
                    sex_pairwise_target = label_to_pair_compare(
                        list(chunks(sex_target, args.slate_num))
                    ).to(sex_target.device)
                    con_pairwise_target = label_to_pair_compare(
                        list(chunks(con_target, args.slate_num))
                    ).to(con_target.device)
                    pos_pairwise_target = label_to_pair_compare(
                        list(chunks(pos_target, args.slate_num))
                    ).to(pos_target.device)

                    #                     prediction = model(text, batch_size = len(target))
                    rank_pred, cls_pred = model(text, batch_size=len(target))

                    vio_cls_loss = loss_function(cls_pred[0], vio_target)
                    sub_cls_loss = loss_function(cls_pred[1], sub_target)
                    sex_cls_loss = loss_function(cls_pred[2], sex_target)
                    con_cls_loss = loss_function(cls_pred[3], con_target)
                    pos_cls_loss = loss_function(cls_pred[4], pos_target)

                    vio_rank_loss = loss_function(rank_pred[0], vio_pairwise_target)
                    sub_rank_loss = loss_function(rank_pred[1], sub_pairwise_target)
                    sex_rank_loss = loss_function(rank_pred[2], sex_pairwise_target)
                    con_rank_loss = loss_function(rank_pred[3], con_pairwise_target)
                    pos_rank_loss = loss_function(rank_pred[4], pos_pairwise_target)

                    vio_pred_digits = [torch.argmax(x).item() for x in cls_pred[0]]
                    sub_pred_digits = [torch.argmax(x).item() for x in cls_pred[1]]
                    sex_pred_digits = [torch.argmax(x).item() for x in cls_pred[2]]
                    con_pred_digits = [torch.argmax(x).item() for x in cls_pred[3]]
                    pos_pred_digits = [torch.argmax(x).item() for x in cls_pred[4]]

                    val_loss = (
                        vio_cls_loss
                        + sub_cls_loss
                        + sex_cls_loss
                        + con_cls_loss
                        + pos_cls_loss
                        + vio_rank_loss
                        + sub_rank_loss
                        + sex_rank_loss
                        + con_rank_loss
                        + pos_rank_loss
                    )

                    # record output from each step
                    val_record = {
                        "val_loss": val_loss,
                        "vio_pred_digits": vio_pred_digits,
                        "sub_pred_digits": sub_pred_digits,
                        "sex_pred_digits": sex_pred_digits,
                        "con_pred_digits": con_pred_digits,
                        "pos_pred_digits": pos_pred_digits,
                        "vio_target": vio_target.tolist(),
                        "sub_target": sub_target.tolist(),
                        "sex_target": sex_target.tolist(),
                        "con_target": con_target.tolist(),
                        "pos_target": pos_target.tolist(),
                    }
                    # collect with a list
                    val_step_outputs.append(val_record)

            print("======== validation end =======")
            # ======== validation end =======#
            avg_val_loss = torch.tensor(
                [x["val_loss"] for x in val_step_outputs]
            ).mean()
            # unpack list of lists --  new,  list(itertools.chain(*list2d))
            val_vio_pred_digits = list(
                itertools.chain(*[x["vio_pred_digits"] for x in val_step_outputs])
            )
            val_sub_pred_digits = list(
                itertools.chain(*[x["sub_pred_digits"] for x in val_step_outputs])
            )
            val_sex_pred_digits = list(
                itertools.chain(*[x["sex_pred_digits"] for x in val_step_outputs])
            )
            val_con_pred_digits = list(
                itertools.chain(*[x["con_pred_digits"] for x in val_step_outputs])
            )
            val_pos_pred_digits = list(
                itertools.chain(*[x["pos_pred_digits"] for x in val_step_outputs])
            )

            val_vio_target = list(
                itertools.chain(*[x["vio_target"] for x in val_step_outputs])
            )
            val_sub_target = list(
                itertools.chain(*[x["sub_target"] for x in val_step_outputs])
            )
            val_sex_target = list(
                itertools.chain(*[x["sex_target"] for x in val_step_outputs])
            )
            val_con_target = list(
                itertools.chain(*[x["con_target"] for x in val_step_outputs])
            )
            val_pos_target = list(
                itertools.chain(*[x["pos_target"] for x in val_step_outputs])
            )

            dev_cls_report_vio = classification_report(
                val_vio_target, val_vio_pred_digits, digits=4
            )
            dev_cls_report_sub = classification_report(
                val_sub_target, val_sub_pred_digits, digits=4
            )
            dev_cls_report_sex = classification_report(
                val_sex_target, val_sex_pred_digits, digits=4
            )
            dev_cls_report_con = classification_report(
                val_con_target, val_con_pred_digits, digits=4
            )
            dev_cls_report_pos = classification_report(
                val_pos_target, val_pos_pred_digits, digits=4
            )

            val_f1_vio = f1_score(val_vio_target, val_vio_pred_digits, average="macro")
            val_f1_sub = f1_score(val_sub_target, val_sub_pred_digits, average="macro")
            val_f1_sex = f1_score(val_sex_target, val_sex_pred_digits, average="macro")
            val_f1_con = f1_score(val_con_target, val_con_pred_digits, average="macro")
            val_f1_pos = f1_score(val_pos_target, val_pos_pred_digits, average="macro")

            logger("dev cls f1:")
            print("dev cls f1: \n")
            val_f1_str = "\t".join(
                "{:.4f}".format(e)
                for e in [val_f1_vio, val_f1_sub, val_f1_sex, val_f1_con, val_f1_pos]
            )
            print(val_f1_str)
            logger(val_f1_str)

            avg_val_f1 = (
                sum([val_f1_vio, val_f1_sub, val_f1_sex, val_f1_con, val_f1_pos]) / 5.0
            )

            # ==================early stopping======================
            if args.dev_run != True:
                early_stopping(
                    val_criteria=avg_val_f1, model=model, path=args.model_save_dir
                )

                if early_stopping.new_best:
                    best_sd = copy.deepcopy(model.state_dict())

                if early_stopping.early_stop:
                    print("Early stopping at {}-th epoch.".format(epoch))
                    break
            # ====================adjust lr========================
        #     lr_adjust = {
        #             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #             10: 5e-7, 15: 1e-7, 20: 5e-8
        #         }
        #     if epoch in lr_adjust.keys():
        #         lr = lr_adjust[epoch]
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr
        #         print('Updating learning rate to {}'.format(lr))

        # =====================test============================#
        print("======== test begins =======")

        logger("======== test result =======")

        # watch out for all evaluation, use a pair of same item, so more process on output
        # load best early stop state dict
        model.load_state_dict(best_sd)
        model.eval()
        test_step_outputs = []
        for idx, (text, target) in enumerate(tqdm(test_loader)):
            vio_target = get_target_column(target, 0).to(to_device)
            sub_target = get_target_column(target, 1).to(to_device)
            sex_target = get_target_column(target, 2).to(to_device)
            con_target = get_target_column(target, 3).to(to_device)
            pos_target = get_target_column(target, 4).to(to_device)

            vio_pairwise_target = label_to_pair_compare(
                list(chunks(vio_target, args.slate_num))
            ).to(vio_target.device)
            sub_pairwise_target = label_to_pair_compare(
                list(chunks(sub_target, args.slate_num))
            ).to(sub_target.device)
            sex_pairwise_target = label_to_pair_compare(
                list(chunks(sex_target, args.slate_num))
            ).to(sex_target.device)
            con_pairwise_target = label_to_pair_compare(
                list(chunks(con_target, args.slate_num))
            ).to(con_target.device)
            pos_pairwise_target = label_to_pair_compare(
                list(chunks(pos_target, args.slate_num))
            ).to(pos_target.device)

            #                     prediction = model(text, batch_size = len(target))
            rank_pred, cls_pred = model(text, batch_size=len(target))

            vio_cls_loss = loss_function(cls_pred[0], vio_target)
            sub_cls_loss = loss_function(cls_pred[1], sub_target)
            sex_cls_loss = loss_function(cls_pred[2], sex_target)
            con_cls_loss = loss_function(cls_pred[3], con_target)
            pos_cls_loss = loss_function(cls_pred[4], pos_target)

            vio_rank_loss = loss_function(rank_pred[0], vio_pairwise_target)
            sub_rank_loss = loss_function(rank_pred[1], sub_pairwise_target)
            sex_rank_loss = loss_function(rank_pred[2], sex_pairwise_target)
            con_rank_loss = loss_function(rank_pred[3], con_pairwise_target)
            pos_rank_loss = loss_function(rank_pred[4], pos_pairwise_target)

            vio_pred_digits = [torch.argmax(x).item() for x in cls_pred[0]]
            sub_pred_digits = [torch.argmax(x).item() for x in cls_pred[1]]
            sex_pred_digits = [torch.argmax(x).item() for x in cls_pred[2]]
            con_pred_digits = [torch.argmax(x).item() for x in cls_pred[3]]
            pos_pred_digits = [torch.argmax(x).item() for x in cls_pred[4]]

            test_loss = (
                vio_cls_loss
                + sub_cls_loss
                + sex_cls_loss
                + con_cls_loss
                + pos_cls_loss
                + vio_rank_loss
                + sub_rank_loss
                + sex_rank_loss
                + con_rank_loss
                + pos_rank_loss
            )

            # record output from each step

            test_record = {
                "test_loss": test_loss,
                "vio_pred_digits": [vio_pred_digits[0]],
                "sub_pred_digits": [sub_pred_digits[0]],
                "sex_pred_digits": [sex_pred_digits[0]],
                "con_pred_digits": [con_pred_digits[0]],
                "pos_pred_digits": [pos_pred_digits[0]],
                "vio_target": [vio_target[0].item()],
                "sub_target": [sub_target[0].item()],
                "sex_target": [sex_target[0].item()],
                "con_target": [con_target[0].item()],
                "pos_target": [pos_target[0].item()],
            }
            # collect with a list
            test_step_outputs.append(test_record)

        # ======== test end =======#
        avg_test_loss = torch.tensor([x["test_loss"] for x in test_step_outputs]).mean()

        test_vio_pred_digits = list(
            itertools.chain(*[x["vio_pred_digits"] for x in test_step_outputs])
        )
        test_sub_pred_digits = list(
            itertools.chain(*[x["sub_pred_digits"] for x in test_step_outputs])
        )
        test_sex_pred_digits = list(
            itertools.chain(*[x["sex_pred_digits"] for x in test_step_outputs])
        )
        test_con_pred_digits = list(
            itertools.chain(*[x["con_pred_digits"] for x in test_step_outputs])
        )
        test_pos_pred_digits = list(
            itertools.chain(*[x["pos_pred_digits"] for x in test_step_outputs])
        )

        test_vio_target = list(
            itertools.chain(*[x["vio_target"] for x in test_step_outputs])
        )
        test_sub_target = list(
            itertools.chain(*[x["sub_target"] for x in test_step_outputs])
        )
        test_sex_target = list(
            itertools.chain(*[x["sex_target"] for x in test_step_outputs])
        )
        test_con_target = list(
            itertools.chain(*[x["con_target"] for x in test_step_outputs])
        )
        test_pos_target = list(
            itertools.chain(*[x["pos_target"] for x in test_step_outputs])
        )

        test_f1_vio = f1_score(test_vio_target, test_vio_pred_digits, average="macro")
        test_f1_sub = f1_score(test_sub_target, test_sub_pred_digits, average="macro")
        test_f1_sex = f1_score(test_sex_target, test_sex_pred_digits, average="macro")
        test_f1_con = f1_score(test_con_target, test_con_pred_digits, average="macro")
        test_f1_pos = f1_score(test_pos_target, test_pos_pred_digits, average="macro")

        logger("test cls f1:")
        print("test cls f1: \n")

        test_f1_str = "\t".join(
            "{:.4f}".format(e)
            for e in [test_f1_vio, test_f1_sub, test_f1_sex, test_f1_con, test_f1_pos]
        )
        print(test_f1_str)
        logger(test_f1_str)

        # after one fold
        folds_f1_collection.append(
            [test_f1_vio, test_f1_sub, test_f1_sex, test_f1_con, test_f1_pos]
        )

    one_cv_average = np.mean(np.array(folds_f1_collection), axis=0)
    logger("One CV average f1:")
    logger("\t".join("{:.4f}".format(e) for e in one_cv_average))
    all_seed_f1_collection.append(one_cv_average)

result_table = np.array(all_seed_f1_collection)

logger("Final result:")
for each_seed_result in result_table:
    logger("\t".join("{:.4f}".format(e) for e in each_seed_result))

logger("Final average result for each aspect:")
aspect_avg = np.mean(result_table, axis=0)
logger("\t".join("{:.4f}".format(e) for e in aspect_avg))
