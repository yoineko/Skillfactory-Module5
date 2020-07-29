import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import pickle


DIR = 'data/'

@profile
def get_names(preds_by_item):
    """
    Функция для возвращения имени товаров
    return - list of names
    """
  
    #names = list(preds_by_item['index'].apply(lambda x: item_title_map[item_reverse_id_map[x]] )) 
    #preds_by_item['index'] = preds_by_item['index'].map(item_reverse_id_map)
    #names = list(preds_by_item['index'].map(item_title_map))
    names = [item_title_map[item_reverse_id_map[x]] for x in list(preds_by_item['index'])]
    return names

#@st.cache
@profile
def read_files(folder_name):
    """
    Функция для чтения файлов + преобразование к  нижнему регистру
    """
    ratings = pd.read_csv(folder_name+'data_clean.csv',low_memory=False) #  names=['asin','reviewerName','userid','itemid'], verbose=True,
    users = ratings.drop_duplicates(subset =['userid'], keep = 'last')[['userid','reviewerName']]
    users['reviewerName'] = users.reviewerName.str.lower()+ ' (' + users.userid.astype('str') + ')'
    return users

@profile
def load_embeddings(file):
    """
    Функция для загрузки векторных представлений
    """
    with open(file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


#Загружаем данные
users = read_files(folder_name=DIR) 
user_id_map = load_embeddings(DIR +'user_id_map_base0.pkl')
item_id_map = load_embeddings(DIR +'item_id_map_base0.pkl')
model_embeddings = load_embeddings(DIR +'myfile_base0.pkl') 
item_title_map = load_embeddings(DIR +'item_title_map_base0.pkl')               

item_reverse_id_map = {value: key for key,value  in item_id_map.items()}

st.header('Рекомендации для выбранного пользователя')

#Объект для кол-ва топ
#topn = 10
topn = st.slider('Задайте количество записей для ТОП :',min_value=1, max_value=10, value=3, step=1)

#Форма для ввода текста
name_user = st.text_input('Шаблон поиска', '')
# name_user = '' 
# name_user ='david waggane'
name_user = name_user.lower()

#Наш поиск по пользователям
output = users[users.reviewerName.str.contains(name_user) > 0]

#Выбор пользователя из списка
option = st.selectbox('Имя пользователя', output['reviewerName'].values, key=output['userid'].values)

#option =  output.iloc[0]['reviewerName'] #.values

#Выводим пользователя
'Ваш выбор: ', option

# #Ищем рекомендации для пользователя
#with st.spinner('Подбираем...'):
#Определяем id пользователя
user = output[output['reviewerName'].values == option].userid.iloc[0]
#Делаем предсказание для пользователя по всему списку продуктов
preds_light = model_embeddings.predict(user_id_map[user], list(item_id_map.values()))
#Выбираем только ТОП N рекомендаций
preds_light = pd.DataFrame(preds_light, columns=['rating']).sort_values(by= 'rating', ascending= False)[:topn].reset_index()

#Выводим рекомендации для пользователя
'ТОП '+ str(topn) +' товары: '
st.write('', get_names(preds_light))
