# _*_ coding: utf-8 _*_

import argparse

from tqdm import tqdm
import itertools
import time
import copy

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

parser.add_argument("--use_gpu_idx", type=int, default=7)

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


logger = ExperimentLog(args.log_save_dir + "record_binary_10CV.txt")

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

        def get_column(matrix, i):
            return torch.stack([row[i] for row in matrix])

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

        def digits_to_binary_attributes(label):
            attr_one = 0  # [<mid, >=mid] - [0, 1]
            attr_two = 0  # [<high, >=high] - [0, 1]
            if label == 0:
                attr_one = 0
                attr_two = 0
            elif label == 1:
                attr_one = 1
                attr_two = 0
            else:
                attr_one = 1
                attr_two = 1
            return (attr_one, attr_two)

        def digits_to_binary_attributes_list(label_list):
            binary_list = [digits_to_binary_attributes(each) for each in label_list]
            attr_one_list = torch.tensor([each[0] for each in binary_list]).to(
                to_device
            )
            attr_two_list = torch.tensor([each[1] for each in binary_list]).to(
                to_device
            )

            return attr_one_list, attr_two_list

        def binary_attributes_to_digits(attr_one, attr_two):
            label = 0
            if attr_one == 0 and attr_two == 0:
                label = 0
            elif attr_one == 1 and attr_two == 0:
                label = 1
            else:
                label = 2
            return label

        def binary_attributes_to_digits_list(list_one, list_two):
            return [
                binary_attributes_to_digits(list_one[i], list_two[i])
                for i in range(len(list_one))
            ]

        class SiameseLSTM(nn.Module):
            def __init__(
                self,
                input_size,
                slate_num,
                output_size,
                class_num,
                hidden_size,
                projection_size,
            ):
                super(SiameseLSTM, self).__init__()

                bsz = 1
                self.direction = 2
                self.input_size = input_size
                self.slate_num = slate_num
                self.output_size = output_size
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

                self.vio_classifier_one = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.sub_classifier_one = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.sex_classifier_one = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.con_classifier_one = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.pos_classifier_one = nn.Linear(
                    self.projection_size, self.class_num
                )

                self.vio_classifier_two = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.sub_classifier_two = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.sex_classifier_two = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.con_classifier_two = nn.Linear(
                    self.projection_size, self.class_num
                )
                self.pos_classifier_two = nn.Linear(
                    self.projection_size, self.class_num
                )

            def forward(self, sent_sem, sent_emo, batch_size=None):
                # print("here")
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

                vio_output_one = self.vio_classifier_one(vio_att_applied)
                sub_output_one = self.sub_classifier_one(sub_att_applied)
                sex_output_one = self.sex_classifier_one(sex_att_applied)
                con_output_one = self.con_classifier_one(con_att_applied)
                pos_output_one = self.pos_classifier_one(pos_att_applied)

                vio_output_two = self.vio_classifier_two(vio_att_applied)
                sub_output_two = self.sub_classifier_two(sub_att_applied)
                sex_output_two = self.sex_classifier_two(sex_att_applied)
                con_output_two = self.con_classifier_two(con_att_applied)
                pos_output_two = self.pos_classifier_two(pos_att_applied)

                final_output_one = (
                    vio_output_one,
                    sub_output_one,
                    sex_output_one,
                    con_output_one,
                    pos_output_one,
                )
                final_output_two = (
                    vio_output_two,
                    sub_output_two,
                    sex_output_two,
                    con_output_two,
                    pos_output_two,
                )

                return (final_output_one, final_output_two)

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

                return {"text": text, "emo": emo, "label": y_label}

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        def my_collate_fn(batch):
            text_batch = [each_item["text"] for each_item in batch]
            emo_batch = [each_item["emo"] for each_item in batch]
            label_batch = [each_item["label"] for each_item in batch]
            data_length = [len(sq) for sq in text_batch]
            # sort from longest to shortest
            text_batch = [
                x
                for _, x in sorted(
                    zip(data_length, text_batch), key=lambda pair: pair[0], reverse=True
                )
            ]
            emo_batch = [
                x
                for _, x in sorted(
                    zip(data_length, emo_batch), key=lambda pair: pair[0], reverse=True
                )
            ]
            # no stack for 5
            label_batch = [
                x
                for _, x in sorted(
                    zip(data_length, label_batch),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ]
            data_length.sort(reverse=True)
            # pad
            text_batch = pad_sequence(text_batch, batch_first=True, padding_value=0)
            emo_batch = pad_sequence(emo_batch, batch_first=True, padding_value=0)
            # pack padded
            text_batch = pack_padded_sequence(text_batch, data_length, batch_first=True)
            emo_batch = pack_padded_sequence(emo_batch, data_length, batch_first=True)
            #     return text_batch, data_length, label_batch
            return text_batch, emo_batch, label_batch

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
            collate_fn=my_collate_fn,
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
                    for idx, (text, emo, target) in enumerate(tepoch):
                        text = text.to(to_device)
                        emo = emo.to(to_device)
                        #                     target = target.to(to_device)
                        vio_target = get_column(target, 0)
                        sub_target = get_column(target, 1)
                        sex_target = get_column(target, 2)
                        con_target = get_column(target, 3)
                        pos_target = get_column(target, 4)

                        vio_target_one, vio_target_two = (
                            digits_to_binary_attributes_list(vio_target)
                        )
                        sub_target_one, sub_target_two = (
                            digits_to_binary_attributes_list(sub_target)
                        )
                        sex_target_one, sex_target_two = (
                            digits_to_binary_attributes_list(sex_target)
                        )
                        con_target_one, con_target_two = (
                            digits_to_binary_attributes_list(con_target)
                        )
                        pos_target_one, pos_target_two = (
                            digits_to_binary_attributes_list(pos_target)
                        )

                        #                     prediction = model(text, batch_size = len(target))
                        pred_one, pred_two = model(text, emo, batch_size=len(target))

                        vio_loss_one = loss_function(pred_one[0], vio_target_one)
                        sub_loss_one = loss_function(pred_one[1], sub_target_one)
                        sex_loss_one = loss_function(pred_one[2], sex_target_one)
                        con_loss_one = loss_function(pred_one[3], con_target_one)
                        pos_loss_one = loss_function(pred_one[4], pos_target_one)

                        vio_loss_two = loss_function(pred_two[0], vio_target_two)
                        sub_loss_two = loss_function(pred_two[1], sub_target_two)
                        sex_loss_two = loss_function(pred_two[2], sex_target_two)
                        con_loss_two = loss_function(pred_two[3], con_target_two)
                        pos_loss_two = loss_function(pred_two[4], pos_target_two)

                        loss = (
                            vio_loss_one
                            + sub_loss_one
                            + sex_loss_one
                            + con_loss_one
                            + pos_loss_one
                            + vio_loss_two
                            + sub_loss_two
                            + sex_loss_two
                            + con_loss_two
                            + pos_loss_two
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
                for idx, (text, emo, target) in enumerate(vepoch):
                    text = text.to(to_device)
                    emo = emo.to(to_device)
                    #                 target = target.to(to_device)
                    vio_target = get_column(target, 0)
                    sub_target = get_column(target, 1)
                    sex_target = get_column(target, 2)
                    con_target = get_column(target, 3)
                    pos_target = get_column(target, 4)

                    vio_target_one, vio_target_two = digits_to_binary_attributes_list(
                        vio_target
                    )
                    sub_target_one, sub_target_two = digits_to_binary_attributes_list(
                        sub_target
                    )
                    sex_target_one, sex_target_two = digits_to_binary_attributes_list(
                        sex_target
                    )
                    con_target_one, con_target_two = digits_to_binary_attributes_list(
                        con_target
                    )
                    pos_target_one, pos_target_two = digits_to_binary_attributes_list(
                        pos_target
                    )

                    pred_one, pred_two = model(text, emo, batch_size=len(target))

                    vio_loss_one = loss_function(pred_one[0], vio_target_one)
                    sub_loss_one = loss_function(pred_one[1], sub_target_one)
                    sex_loss_one = loss_function(pred_one[2], sex_target_one)
                    con_loss_one = loss_function(pred_one[3], con_target_one)
                    pos_loss_one = loss_function(pred_one[4], pos_target_one)

                    vio_loss_two = loss_function(pred_two[0], vio_target_two)
                    sub_loss_two = loss_function(pred_two[1], sub_target_two)
                    sex_loss_two = loss_function(pred_two[2], sex_target_two)
                    con_loss_two = loss_function(pred_two[3], con_target_two)
                    pos_loss_two = loss_function(pred_two[4], pos_target_two)

                    vio_pred_digits_one = [torch.argmax(x).item() for x in pred_one[0]]
                    sub_pred_digits_one = [torch.argmax(x).item() for x in pred_one[1]]
                    sex_pred_digits_one = [torch.argmax(x).item() for x in pred_one[2]]
                    con_pred_digits_one = [torch.argmax(x).item() for x in pred_one[3]]
                    pos_pred_digits_one = [torch.argmax(x).item() for x in pred_one[4]]

                    vio_pred_digits_two = [torch.argmax(x).item() for x in pred_two[0]]
                    sub_pred_digits_two = [torch.argmax(x).item() for x in pred_two[1]]
                    sex_pred_digits_two = [torch.argmax(x).item() for x in pred_two[2]]
                    con_pred_digits_two = [torch.argmax(x).item() for x in pred_two[3]]
                    pos_pred_digits_two = [torch.argmax(x).item() for x in pred_two[4]]

                    vio_pred_digits = binary_attributes_to_digits_list(
                        vio_pred_digits_one, vio_pred_digits_two
                    )
                    sub_pred_digits = binary_attributes_to_digits_list(
                        sub_pred_digits_one, sub_pred_digits_two
                    )
                    sex_pred_digits = binary_attributes_to_digits_list(
                        sex_pred_digits_one, sex_pred_digits_two
                    )
                    con_pred_digits = binary_attributes_to_digits_list(
                        con_pred_digits_one, con_pred_digits_two
                    )
                    pos_pred_digits = binary_attributes_to_digits_list(
                        pos_pred_digits_one, pos_pred_digits_two
                    )

                    val_loss = (
                        vio_loss_one
                        + sub_loss_one
                        + sex_loss_one
                        + con_loss_one
                        + pos_loss_one
                        + vio_loss_two
                        + sub_loss_two
                        + sex_loss_two
                        + con_loss_two
                        + pos_loss_two
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
        for idx, (text, emo, target) in enumerate(tqdm(test_loader)):
            text = text.to(to_device)
            emo = emo.to(to_device)

            vio_target = get_column(target, 0)
            sub_target = get_column(target, 1)
            sex_target = get_column(target, 2)
            con_target = get_column(target, 3)
            pos_target = get_column(target, 4)

            vio_target_one, vio_target_two = digits_to_binary_attributes_list(
                vio_target
            )
            sub_target_one, sub_target_two = digits_to_binary_attributes_list(
                sub_target
            )
            sex_target_one, sex_target_two = digits_to_binary_attributes_list(
                sex_target
            )
            con_target_one, con_target_two = digits_to_binary_attributes_list(
                con_target
            )
            pos_target_one, pos_target_two = digits_to_binary_attributes_list(
                pos_target
            )

            pred_one, pred_two = model(text, emo, batch_size=len(target))

            vio_loss_one = loss_function(pred_one[0], vio_target_one)
            sub_loss_one = loss_function(pred_one[1], sub_target_one)
            sex_loss_one = loss_function(pred_one[2], sex_target_one)
            con_loss_one = loss_function(pred_one[3], con_target_one)
            pos_loss_one = loss_function(pred_one[4], pos_target_one)

            vio_loss_two = loss_function(pred_two[0], vio_target_two)
            sub_loss_two = loss_function(pred_two[1], sub_target_two)
            sex_loss_two = loss_function(pred_two[2], sex_target_two)
            con_loss_two = loss_function(pred_two[3], con_target_two)
            pos_loss_two = loss_function(pred_two[4], pos_target_two)

            vio_pred_digits_one = [torch.argmax(x).item() for x in pred_one[0]]
            sub_pred_digits_one = [torch.argmax(x).item() for x in pred_one[1]]
            sex_pred_digits_one = [torch.argmax(x).item() for x in pred_one[2]]
            con_pred_digits_one = [torch.argmax(x).item() for x in pred_one[3]]
            pos_pred_digits_one = [torch.argmax(x).item() for x in pred_one[4]]

            vio_pred_digits_two = [torch.argmax(x).item() for x in pred_two[0]]
            sub_pred_digits_two = [torch.argmax(x).item() for x in pred_two[1]]
            sex_pred_digits_two = [torch.argmax(x).item() for x in pred_two[2]]
            con_pred_digits_two = [torch.argmax(x).item() for x in pred_two[3]]
            pos_pred_digits_two = [torch.argmax(x).item() for x in pred_two[4]]

            vio_pred_digits = binary_attributes_to_digits_list(
                vio_pred_digits_one, vio_pred_digits_two
            )
            sub_pred_digits = binary_attributes_to_digits_list(
                sub_pred_digits_one, sub_pred_digits_two
            )
            sex_pred_digits = binary_attributes_to_digits_list(
                sex_pred_digits_one, sex_pred_digits_two
            )
            con_pred_digits = binary_attributes_to_digits_list(
                con_pred_digits_one, con_pred_digits_two
            )
            pos_pred_digits = binary_attributes_to_digits_list(
                pos_pred_digits_one, pos_pred_digits_two
            )

            test_loss = (
                vio_loss_one
                + sub_loss_one
                + sex_loss_one
                + con_loss_one
                + pos_loss_one
                + vio_loss_two
                + sub_loss_two
                + sex_loss_two
                + con_loss_two
                + pos_loss_two
            )

            # record output from each step

            test_record = {
                "test_loss": test_loss,
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
