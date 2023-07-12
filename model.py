import torch
import torch.nn as nn
from gensim.models import Word2Vec
import torchvision.models as models
from data import get_data, get_dataloader, process_data
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ResNet-18 + GRU
class MultiModalTransformer_1(nn.Module):
    def __init__(self, vocab_len, embed_dim, hidden_size, num_classes):
        super(MultiModalTransformer_1, self).__init__()

        # 图像特征提取器（可以使用预训练的CNN模型，如ResNet等）
        self.image_encoder = models.resnet18(pretrained=True)
        # 固定resnet-18参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.att_weights = nn.Linear(in_features=1000, out_features=hidden_size*2)

        # 文本特征提取器（可以使用预训练的Transformer模型）
        self.embedding = nn.Embedding(vocab_len, embed_dim)
        self.text_encoder = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

        # 全连接层
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, image, text):
        image_features = self.image_encoder(image)  # 图像特征形状: [batch_size, 1000]
        image_features = self.tanh(self.att_weights(image_features))  # [batch_size, h*2]

        embedded_text = self.embedding(text)
        text_features, _ = self.text_encoder(embedded_text)  # 文本特征形状: [batch_size, sequence_length, hidden_size*2]
        text_features = self.tanh(text_features)

        att_score = torch.einsum('bj,bij->bi', image_features, text_features)  # [b, max_len]
        att_score = self.softmax(att_score)

        context_vector = torch.einsum('bij, bi->bj', text_features, att_score)  # [b, h*2]

        # 对文本特征进行池化降维
        # pooled_text_features = torch.mean(text_features, dim=1)  # 形状: [batch_size, hidden_size*2]


        # 将图像特征和文本特征在特征维度上进行拼接
        # combined_features = torch.cat((image_features, pooled_text_features), dim=1)  # 形状: [batch_size,2*hidden_size]

        # 进行情感分类
        output = self.fc(context_vector)

        return output


# VGGNet-16  GRU
class MultiModalTransformer_2(nn.Module):
    def __init__(self, vocab_len, embed_dim, hidden_size, num_classes):
        super(MultiModalTransformer_2, self).__init__()

        # 图像特征提取器（使用预训练的VGGNet模型）
        self.image_encoder = models.vgg16(pretrained=True)
        # 固定VGGNet参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.att_weights = nn.Linear(in_features=2048, out_features=hidden_size * 2)
        self.max_pool = nn.MaxPool2d(3)

        # 文本特征提取器（可以使用预训练的Transformer模型）
        self.embedding = nn.Embedding(vocab_len, embed_dim)
        self.text_encoder = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        # 全连接层
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 融合后的特征维度为4096 + hidden_size

    def forward(self, image, text):
        image_features = self.image_encoder.features(image)  # [batch_size, 512, 7, 7]
        image_features = self.max_pool(image_features)  # [batch_size, 512, 2, 2]
        image_features = image_features.view(image_features.size(0), -1)  # [batch_size, 2048]
        att_vector = self.tanh(self.att_weights(image_features))  # [batch_size, 256]

        embedded_text = self.embedding(text)
        text_features, _ = self.text_encoder(embedded_text)
        text_features = self.tanh(text_features)  # [batch_size, 30, 256]

        att_score = torch.einsum('BJ,BIJ->BI', att_vector, text_features)
        att_score = self.softmax(att_score)  # [16, 30]

        context_vector = torch.einsum('BIJ,BI->BJ', text_features, att_score)  # [batch_size, 256]

        # 进行情感分类
        output = self.fc(context_vector)
        return output


# resnet-18 (word2vec)Embedding gru
class MultiModalTransformer_3(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MultiModalTransformer_3, self).__init__()

        # 图像特征提取器（使用预训练的ResNet-18模型）
        self.image_encoder = models.resnet18(pretrained=True)
        # 固定ResNet-18参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.att_weights = nn.Linear(in_features=1000, out_features=hidden_size * 2)

        # 文本特征提取器（使用预训练的Word2Vec模型）
        word2vec_model = Word2Vec.load('word2vec.model')
        vocab_size, embed_dim = word2vec_model.wv.vectors.shape
        print(vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec_model.wv.vectors))
        self.embedding.weight.requires_grad = False

        # 文本特征提取器（使用gru模型）
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        # 全连接层
        self.out_layer = nn.Linear(hidden_size*4, num_classes)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, image=None, text=None):
        if image is not None and text is not None:
            image_features = self.image_encoder(image)  # 图像特征形状: [batch_size, 1000]
            image_features = self.tanh(self.att_weights(image_features))  # [batch_size, h*2]

            embedded_text = self.embedding(text)
            text_features, _ = self.gru(embedded_text)  # 文本特征形状: [batch_size, sequence_length, hidden_size*2]
            text_features = self.tanh(text_features)[:, -1, :]

            # cat_features = torch.cat([image_features, text_features[:, -1, :]], dim=1)
            # output = self.out_layer(cat_features)
            fused_features = image_features * 0.6 + text_features * 0.4
            output = self.fc(fused_features)
            return output

        elif text is None:
            # 只使用图像输入分类
            image_features = self.image_encoder(image)  # 图像特征形状: [batch_size, 1000]
            image_features = self.tanh(self.att_weights(image_features))  # [batch_size, h*2]
            output = self.fc(image_features)

        elif image is None:
            # 只使用文本输入分类
            embedded_text = self.embedding(text)
            text_features, _ = self.gru(embedded_text)  # 文本特征形状: [batch_size, sequence_length, hidden_size*2]
            text_features = self.tanh(text_features)
            context_vector = text_features[:, -1, :]
            output = self.fc(context_vector)

        else:
            print('Must have image or text Input !')

        return output


def train(args):
    train_data, test_data = get_data()
    train_data, test_data, vocab = process_data(train_data, test_data)
    vocab_len = len(vocab) + 1
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(train_data, test_data)

    if args.model == 1:
        model = MultiModalTransformer_1(vocab_len, embed_dim=64, hidden_size=128, num_classes=3)
    elif args.model == 2:
        model = MultiModalTransformer_2(vocab_len, embed_dim=64, hidden_size=128, num_classes=3)
    elif args.model == 3:
        model = MultiModalTransformer_3(hidden_size=128, num_classes=3)
    else:
        print('waiting-----')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_rate = 0.0
    print('='*25 + ' START TRAINING ' + '='*25)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        start = time.time()
        for guids, tags, images, texts in train_dataloader:
            optimizer.zero_grad()

            if args.image_only:
                outputs = model(image=images, text=None)
            elif args.text_only:
                outputs = model(image=None, text=texts)
            else:
                outputs = model(images, texts)

            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*len(guids)
            pred = torch.max(outputs, 1)[1]
            total += len(guids)
            correct += (pred == tags).sum()

            target_list.extend(tags.tolist())
            pred_list.extend(pred.tolist())

        train_loss /= total

        print('[EPOCH {:02d}]'.format(epoch + 1), end=' ')
        print('Train Loss:{:.4f}'.format(train_loss), end=' ')
        rate = correct/total*100
        print('Accuracy Rate:{:.2f}%'.format(rate), end='  ')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        target_list = []
        pred_list = []

        with torch.no_grad():
            for guids, tags, images, texts in val_dataloader:
                if args.image_only:
                    outputs = model(image=images, text=None)
                elif args.text_only:
                    outputs = model(image=None, text=texts)
                else:
                    outputs = model(images, texts)

                loss = criterion(outputs, tags)
                val_loss += loss.item()*len(guids)
                pred = torch.max(outputs, 1)[1]
                total += len(guids)
                correct += (pred == tags).sum()

                target_list.extend(tags.tolist())
                pred_list.extend(pred.tolist())

        val_loss /= total
        print('Valid Loss:{:.4f}'.format(val_loss), end=' ')
        rate = correct/total*100
        print('Accuracy Rate:{:.2f}%'.format(rate), end='  ')
        end = time.time()
        print('time:{:.2f}s'.format(end-start))

        if rate > best_rate:
            best_rate = rate
            print('BEST RATE:{:.2f}%'.format(rate))
            print('model has been saved in model.pth')
            torch.save(model.state_dict(), 'model.pth')

    print('='*25 + ' TRAINING END ' + '='*25)


def test(args):
    train_data, test_data = get_data()
    train_data, test_data, vocab = process_data(train_data, test_data)
    vocab_len = len(vocab) + 1
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(train_data, test_data)

    if args.model == 1:
        model = MultiModalTransformer_1(vocab_len, embed_dim=64, hidden_size=128, num_classes=3)
    elif args.model == 2:
        model = MultiModalTransformer_2(vocab_len, embed_dim=64, hidden_size=128, num_classes=3)
    elif args.model == 3:
        model = MultiModalTransformer_3(hidden_size=128, num_classes=3)
    else:
        print('error model waiting-----')

    model.load_state_dict(torch.load('model.pth'))

    guids = []
    tags = []
    model.eval()

    for guid, tag, image, text in test_dataloader:
        image = image.to(device)
        text = text.to(device)

        if args.text_only:
            out = model(image=None, text=text)
        elif args.image_only:
            out = model(image=image, text=None)
        else:
            out = model(image=image, text=text)

        pred = torch.max(out, 1)[1]
        guids.extend(guid)
        tags.extend(pred.tolist())

    pred_mapped = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    with open('test_with_label.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guids, tags):
            f.write(f'{guid},{pred_mapped[pred]}\n')
        f.close()

    print('prediction has been saved to test_with_label.txt')