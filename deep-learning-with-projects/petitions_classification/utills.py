import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from konlpy.tag import Okt  # 형태소 분석기 한국어 토크나이징 지원, Okt(Twitter)
from numpy.random import RandomState
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import torchtext # pytorch에서 자연어 처리를 지원하는 유요한 라이브러리.
from torchtext.data import Field #텍스트에 대한 dataLodader, 추상화 기능 및 Iterator 제공. Field는 토크나이징 및 단어장 생성 지원.
from torchtext.data import TabularDataset  # 데이터를 읽어, 데이터 생성을 지원.
from torchtext.vocab import Vectors  # 임베딩 벡터를 생성하기 위한 클래서
from torchtext.data import BucketIterator  # 데이터 셋에서 batch 만큼 데이터를 로드하는 Iterator 지원.


def _crawling(crawling_path):
    result = pd.DataFrame()

    for i in range(584274, 595226):
        URL = "http://www1.president.go.kr/petitions/" + str(i)

        response = requests.get(URL)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        title = soup.find('h3', class_='petitionsView_title')
        count = soup.find('span', class_='counter')

        for content in soup.select('div.petitionsView_write > div.View_write'):
            content

        a = []
        for tag in soup.select('ul.petitionsView_info_list > li'):
            a.append(tag.contents[1])

        if len(a) != 0:
            df1 = pd.DataFrame({'start': [a[1]],
                                'end': [a[2]],
                                'category': [a[0]],
                                'count': [count.text],
                                'title': [title.text],
                                'content': [content.text.strip()[0:13000]]
                                })

            result = pd.concat([result, df1])
            result.index = np.arange(len(result))

        if i % 60 == 0:
            print("Sleep 90seconds. Count:" + str(i)
                  + ",  Local Time:" + time.strftime('%Y-%m-%d', time.localtime(time.time()))
                  + " " + time.strftime('%X', time.localtime(time.time()))
                  + ",  Data Length:" + str(len(result)))
            time.sleep(90)
    print(result.shape)
    df = result
    df.to_csv(crawling_path, index=False, encoding='utf-8-sig')

    return df


def _remove_white_space(text):
    text = re.sub(r'[\t\r\n\f\v]', ' ', str(text))
    return text


def _remove_special_char(text):
    text = re.sub('[^ ㄱ-ㅣ가-힣 0-9]+', ' ', str(text))
    return text


def _preprocessing(df):
    df.title = df.title.apply(_remove_white_space)
    df.title = df.title.apply(_remove_special_char)

    df.content = df.content.apply(_remove_white_space)
    df.content = df.content.apply(_remove_special_char)

    return df


def _token(df, token_path):
    okt = Okt()
    df['title_token'] = df.title.apply(okt.morphs)
    df['content_token'] = df.content.apply(okt.nouns)
    df['token_final'] = df.title_token + df.content_token
    df['count'] = df['count'].replace({',': ''}, regex=True).apply(lambda x: int(x))
    df['label'] = df['count'].apply(lambda x: 'Yes' if x >= 1000 else 'No')
    df_drop = df[['token_final', 'label']]

    df_drop.to_csv(token_path, index=False, encoding='utf-8-sig')

    return df_drop


def _embedding(df_drop, emb_path):
    # [단어 임베딩] # 형태소로 나눈 제목과, 명사로 나눈 내용을 합해서, 각 단어마다 one-hot-vecotr(100,)로 만들고 이를 (100,100)에 Linear projection
    embedding_model = Word2Vec(df_drop['token_final'],
                               sg=1,  # skip-gram, 0: CBOW
                               vector_size=100,
                               window=2,
                               min_count=1,
                               workers=4
                               )
    # """
    # [임베딩 모델 저장 및 로드]
    embedding_model.wv.save_word2vec_format(emb_path)  # 모델 저장
    loaded_model = KeyedVectors.load_word2vec_format(emb_path)  # 모델 로드

    model_result = loaded_model.most_similar("음주운전")
    print(model_result)


def _train_val_data(df_drop, train_path, val_path):
    rng = RandomState() # 난수 생성.

    # pandas.sample() : frac=%만큼의 sample 뽑음.
    # pandas.loc[] : train에서 뽑힌 행이 아닌 것들의 데이터 산출.
    tr = df_drop.sample(frac=0.8, random_state=rng) # 데이터 80% train 사용. dataFrame 구조('token_final', 'label' 구조)
    val = df_drop.loc[~df_drop.index.isin(tr.index)] #나머지 사용.

    tr.to_csv(train_path, index=False, encoding='utf-8-sig')
    val.to_csv(val_path, index=False, encoding='utf-8-sig')


def make_data(crawling, crawling_path, token_path, train_path, val_path, emb_path):
    if crawling:
        df = _crawling(crawling_path)
    else:
        df = pd.read_csv(crawling_path) #(

    df = _preprocessing(df)
    df_drop = _token(df, token_path)
    #df_drop = pd.read_csv(token_path)
    _embedding(df_drop, emb_path)
    _train_val_data(df_drop, train_path, val_path)

    return


def _tokenizer(text): # Field 클래스에는 ['토큰1', '토큰2', ..] 형태로 입력해야 함.
    text = re.sub('[\[\]\']', '', str(text))
    text = text.split(', ')
    return text


def load_data(embedding_path, device):
    TEXT = Field(tokenize=_tokenizer)  # 데이터 셋의 'token_final', 입력에 사용하기 위해
    LABEL = Field(sequential=False)  # 데이터 셋의 'label' 사용하기 위해.

    # [데이터 불러오기]
    # csv 형식의 train,val 데이터를 읽어 TEXT, LABEL 형식을 가진 데이터셋을 생성함.
    train, validation = TabularDataset.splits(
        path='data/',
        train='train.csv',
        validation='validation.csv',
        format='csv',
        fields=[('text', TEXT), ('label', LABEL)],
        skip_header=True
    )

    print("Train:", train[0].text, train[0].label)
    print("Validation:", validation[0].text, validation[0].label)

    # [단어장 및 DataLoader 정의]

    vectors = Vectors(name=embedding_path)  # 사전에 훈련된(?) 임베딩 벡터를 저장. #총 34506개의 100차원 임베딩 벡터 가지고 있음.

    TEXT.build_vocab(train, vectors=vectors, min_freq=1, max_size=None)  # trian data 단어장 생성, 임베딩 벡터 값으로 초기화.
    LABEL.build_vocab(train)

    vocab = TEXT.vocab

    # batch 만큼 로드하여 생성함.
    train_iter, validation_iter = BucketIterator.splits(
        datasets=(train, validation),
        batch_size=8,
        device=device,
        sort=False
    )

    print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))

    return train_iter, validation_iter, vocab

