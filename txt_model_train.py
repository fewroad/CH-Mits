import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'

import pickle

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel

from config import get_config
from data import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    Transformer_Attention, Transformer_CNN_RNN_Attention, SKNet_Model, SKNet_LSTM_Model, SKNet_LSTM_Attention_Model,\
    SKNetandLSTM_Model


TEXT_MODEL_CONFIG = {
    'bert': {
        'tokenizer_name': 'bert-base-uncased',
        'model_name': 'bert-base-uncased',
        'tokenizer_kwargs': {},
    },
    'roberta': {
        'tokenizer_name': 'roberta-base',
        'model_name': 'roberta-base',
        'tokenizer_kwargs': {'add_prefix_space': True},
    },
    'chi_bert': {
        'tokenizer_name': 'bert-base-chinese',
        'model_name': 'bert-base-chinese',
        'tokenizer_kwargs': {},
    },
    'ernie': {
        'tokenizer_name': 'nghuyong/ernie-3.0-base-zh',
        'model_name': 'nghuyong/ernie-3.0-base-zh',
        'tokenizer_kwargs': {},
    },
}


def build_text_backbone(model_name):
    if model_name not in TEXT_MODEL_CONFIG:
        raise ValueError(f'unknown model: {model_name}')

    model_config = TEXT_MODEL_CONFIG[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['tokenizer_name'],
        **model_config['tokenizer_kwargs'],
    )
    base_model = AutoModel.from_pretrained(model_config['model_name'])
    input_size = getattr(base_model.config, 'hidden_size', 768)
    return tokenizer, base_model, input_size


class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        # Create model
        self.tokenizer, base_model, self.input_size = build_text_backbone(args.model_name)
        self.model_save_path = f'{args.model_name}_{args.method_name}.pkl'
        # Operate the method
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_attention':
            self.Mymodel = Transformer_CNN_RNN_Attention(base_model, args.num_classes, is_val=None)
        elif args.method_name == 'sknet':
            self.Mymodel = SKNet_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'sknet_lstm':
            self.Mymodel = SKNet_LSTM_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'sknet_lstm_attention':
            self.Mymodel = SKNet_LSTM_Attention_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'sknetandlstm':
            self.Mymodel = SKNetandLSTM_Model(base_model, args.num_classes, self.input_size)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0

        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets.long())
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)
        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        # Turn on the eval mode
        self.Mymodel.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets.long())

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    def run(self):
        # Print the parameters of model
        # for name, layer in self.Mymodel.named_parameters(recurse=True):
        # print(name, layer.shape, sep=" ")

        train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = float('inf'), 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                with open(self.model_save_path, "wb") as file:
                    pickle.dump(self.Mymodel, file)
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))


        # with open('textcnn_lstm_attention.pkl', "rb") as file:
        #     model = pickle.load(file)
        #     model.eval()
        #     for inputs, targets in tqdm(test_dataloader, disable=self.args.backend, ascii=' >='):
        #         inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        #         output = model(inputs,is_val='s')

        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        self.logger.info('model saved: {}'.format(self.model_save_path))

        # Draw the training process
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('acc.png')

        plt.plot(l_epo, l_teloss)
        plt.ylabel('test-loss')
        plt.xlabel('epoch')
        plt.savefig('teloss.png')

        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.savefig('trloss.png')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Niubility(args, logger)
    nb.run()
