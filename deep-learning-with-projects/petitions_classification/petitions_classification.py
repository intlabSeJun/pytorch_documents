import torch
import torch.optim as optim
import torch.nn.functional as F
from utills import make_data, load_data
from model import TextCNN


def train(model, device, train_itr, optimizer):
    model.train()
    corrects, train_loss = 0.0, 0

    for batch in train_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)

        optimizer.zero_grad()
        logit = model(text)

        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    train_loss /= len(train_itr.dataset)
    accuracy = 100.0 * corrects / len(train_itr.dataset)

    return train_loss, accuracy


def evaluate(model, device, itr):
    model.eval()
    corrects, test_loss = 0.0, 0

    for batch in itr:
        text = batch.text
        target = batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)

        logit = model(text)
        loss = F.cross_entropy(logit, target)

        test_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    test_loss /= len(itr.dataset)
    accuracy = 100.0 * corrects / len(itr.dataset)

    return test_loss, accuracy


crawling = False  # crawing 할지 여부.
predefined_data = True # 이미 csv 형태도 모든 데이터가 저장되어 있는지 여부.
crawling_path = 'data/crawling.csv'
token_path = 'data/token_final.csv'
train_path = 'data/train.csv'
val_path = 'data/validation.csv'
embedding_path = 'data/petitions_tokens_w2v'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_test_acc = -1

if __name__ == '__main__':
    if predefined_data == False:
        make_data(crawling, crawling_path, token_path, train_path, val_path, embedding_path)  # csv 형태로 데이터 생성.

    train_iter, validation_iter, vocab = load_data(embedding_path, device)
    # 여기까지가 청화대 크롤링 한후, Word2Vec 이용해 임베딩 벡터 생성하고 이를 train,val로 나누어서 data lodaer 만듦.

    model = TextCNN(vocab, 100, 10, [3, 4, 5], 2).to(device) # filter가 문장을 스캔하면서 문맥적 의미를 파악(분류에서는 RNN보다 좋다)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 3 + 1):
        tr_loss, tr_acc = train(model, device, train_iter, optimizer)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))

        val_loss, val_acc = evaluate(model, device, validation_iter)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))

        if val_acc > best_test_acc:
            best_test_acc = val_acc

            print("model saves at {} accuracy".format(best_test_acc))
            torch.save(model.state_dict(), "TextCNN_Best_Validation")

        print('-----------------------------------------------------------------------------')#"""
