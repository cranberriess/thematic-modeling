# thematic-modeling

## 1.	Обоснование выбранных алгоритма тематического моделирования и библиотеки, используемой для реализации.
Для тематического моделирования в данной лабораторной работе выберем Латентное Размещение Дирихле (Latent Dirichlet Allocation, LDA) - применяемая в машинном обучении и информационном поиске порождающая модель, позволяющая объяснять результаты наблюдений с помощью неявных групп, благодаря чему возможно выявление причин сходства некоторых частей данных. 
Латентное распределение Дирихле часто используется для моделирования тем на основе содержания, что в основном означает изучение категорий из неклассифицированного текста. Ранее мы получили статьи, относящиеся к одной категории искусственного интеллекта, их можно объединить в несколько тем. Причем объединение должно происходить на основе содержания с помощью определенных ключевых слов, так как предварительно при веб-скраппинге темы выделены не были. Для решения этой задачи и используется алгоритм LDA. Причем существуют встроенные библиотеки, с помощью которых можно не только разделить статьи на различные темы, но и визуализировать результат для наибольшей наглядности. <br>
В ходе данной лабораторной работы будут использоваться следующие библиотеки:
•	pandas – библиотека для работы с данными (датафреймом)
•	numpy – библиотека для математических вычислений
•	pymorphy2 - морфологический анализатор для русского и английского языков
•	nltk – библиотека для любой задачи обработки естественного языка
•	genism - библиотека для тематического моделирования, индексирования документов, поиска по сходству и других функций обработки естественного языка
•	pyLDAvis – модуль для визуализации работы алгоритма LDA
•	matplotlib – библиотека для визуализации полученных данных
•	wordcloud – пакет для создания облака слов по темам
 
## 2.	Описание предварительной обработки текстов (порядок действий, описание используемых функций).
К предварительной обработке текстов отнесем следующие этапы:
1)	Удаление стоп-слов:
Импортируем библиотеку nltk, скачаем стоп-слова для английского языка, внесем слова 'it’s','you’re', '\\xa', так как они содержат символы, неизвестные для встроенных слов. В качестве проверки выведем список стоп-слов.
2)	Удаление пунктуации:
Импортируем библиотеку pymorphy2, а также модуль MorphAnalyzer. Далее присвоим переменной patterns строку с возможными знаками препинания, а именно "[А-Яа-я0-9!#$%&'()*+,./:;<=>?@[\]«»^_`{|}~—\"’\-]+", внесём также кириллицу, так как все статьи, используемые в данной лабораторной работе содержат только латиницу. 
3)	Лемматизация:
Лемматизация - процесс приведения слова к его нормальной (словарной) форме, он необходим, так как слова в статьях встречаются в разных падежах и склонениях. Лемматизацию будем использовать в функции lemmatize, которая принимает в качестве аргумента текст и выводит токены. На этом этапе функция очистит токены от пунктуации и стоп-слов, приведет слово к его словарной форме, а также обработает исключения в случае ошибки.
После предварительной обработки текста необходимо выделить N-граммы - словосочетание, имеющее признаки синтаксически и семантически целостной единицы, в котором выбор одного из компонентов осуществляется по смыслу, а выбор второго зависит от выбора первого. Для выполнения этого этапа импортируем библиотеку gensim и с помощью Phrases создадим биграммы и триграммы на основе корпуса.<br>
Далее нужно создать словаря и сделать подсчёт когерентности для различного количества тем, чтобы определить оптимальное количество тем, на которые можно разбить все статьи. На этом этапе используем два модуля LdaMulticore и CoherenceModel для функции compute_coherence_values, которая принимает следующие параметры в качестве аргументов: словарь, корпус, список текста, максимальное количество тем, список LDA моделей и когерентности, соответствующие модели LDA с количеством тем. Для наглядности визуализируем согласованность и количество тем, чтобы определить их оптимальное количество.
## 3.	Код скрипта с комментариями основных функций.
```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
Загрузка коллекции документов
```
import pandas as pd
path = "/content/drive/MyDrive/data.csv"
df = pd.read_csv(path, sep = ',')
```
 
Предварительная обработка текстов (удаление стоп-слов, удаление пунктуации, лемматизация)
```
!pip install pymorphy2
!pip install nltk
import re
from pymorphy2 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords

patterns = "[А-Яа-я0-9!#$%&'()*+,./:;<=>?@[\]«»^_`{|}~—\"’\-]+"
nltk.download('stopwords')
stopwords_en = stopwords.words('english')
stopwords_en.extend(['it’s','you’re', '\\xa'])
print(stopwords_en)
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            if token not in stopwords_en:
                tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None

data_ready = df['Article'].apply(lemmatize)
data_ready
```

Разбиение на биграммы и триграммы
```
!pip install gensim
text_clean = []
for index, row in df.iterrows():
        text_clean.append(row['Article'].split())
from gensim.models import Phrases

bigram = Phrases(text_clean)
trigram = Phrases(bigram[text_clean])

for idx in range(len(text_clean)):
    for token in bigram[text_clean[idx]]:
        if '_' in token:
            text_clean[idx].append(token)
    for token in trigram[text_clean[idx]]:
        if '_' in token:
            text_clean[idx].append(token)
```
![image](https://github.com/cranberriess/thematic-modeling/assets/105839329/ace18fcf-e554-418a-b823-675e56b7758f)<br>
Создание словаря частот:
```
from gensim.corpora import Dictionary
from pprint import pprint
import gensim.corpora as corpora
import gensim, spacy, logging, warnings
!pip install pyLDAvis

id2word = corpora.Dictionary(data_ready)
corpus = [id2word.doc2bow(text) for text in data_ready]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=1,
                                           alpha='symmetric',
                                           per_word_topics=True)
[(0,
  '0.008*"object" + 0.007*"would" + 0.006*"data" + 0.006*"images" + '
  '0.006*"openai" + 0.006*"use" + 0.006*"api" + 0.005*"also" + 0.005*"able" + '
  '0.005*"using"'),
 (1,
  '0.023*"data" + 0.015*"ai" + 0.007*"researchers" + 0.006*"used" + '
  '0.006*"model" + 0.006*"new" + 0.005*"learning" + 0.005*"generative" + '
  '0.004*"use" + 0.004*"also"'),
 (2,
  '0.015*"images" + 0.012*"training" + 0.011*"data" + 0.010*"class" + '
  '0.009*"one" + 0.008*"learning" + 0.007*"algorithm" + 0.007*"signal" + '
  '0.006*"group" + 0.006*"classifier"'),
 (3,
  '0.005*"redis" + 0.004*"data" + 0.003*"keys" + 0.003*"ai" + 0.003*"like" + '
  '0.003*"form" + 0.002*"digital" + 0.002*"cluster" + 0.002*"code" + '
  '0.002*"user"'),
 (4,
  '0.019*"ai" + 0.010*"data" + 0.007*"researchers" + 0.005*"learning" + '
  '0.004*"channels" + 0.003*"privacy" + 0.003*"accuracy" + 0.003*"fsl" + '
  '0.003*"intelligence" + 0.003*"power"'),
 (5,
  '0.033*"ai" + 0.015*"power" + 0.013*"data" + 0.007*"fashion" + '
  '0.007*"generative" + 0.005*"new" + 0.005*"learning" + 0.004*"cost" + '
  '0.004*"majority" + 0.004*"privacy"')]
```
Подсчёт когерентности для различного количества тем:
```
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaMulticore(corpus=corpus,id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=2, limit=10, step=2)

import matplotlib.pyplot as plt
limit=10; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Количество тем")
plt.ylabel("Согласованность")
plt.legend(("coherence_values"), loc='best')
plt.show()
```
Создание нового словаря частот
```
id2word = corpora.Dictionary(data_ready)
corpus = [id2word.doc2bow(text) for text in data_ready]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=1,
                                           alpha='symmetric',
                                           per_word_topics=True)
pprint(lda_model.print_topics())
WARNING:gensim.models.ldamodel:too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
[(0,
  '0.010*"data" + 0.008*"ai" + 0.007*"images" + 0.006*"used" + '
  '0.006*"researchers" + 0.005*"object" + 0.005*"use" + 0.005*"would" + '
  '0.005*"neural" + 0.005*"networks"'),
 (1,
  '0.025*"ai" + 0.021*"data" + 0.009*"power" + 0.007*"learning" + '
  '0.007*"generative" + 0.006*"new" + 0.005*"model" + 0.004*"team" + '
  '0.004*"researchers" + 0.004*"models"'),
 (2,
  '0.011*"data" + 0.009*"class" + 0.008*"images" + 0.008*"training" + '
  '0.006*"learning" + 0.006*"one" + 0.005*"model" + 0.005*"signal" + '
  '0.005*"algorithm" + 0.005*"classifier"'),
 (3,
  '0.012*"ai" + 0.007*"data" + 0.004*"fashion" + 0.004*"privacy" + '
  '0.003*"redis" + 0.003*"edge" + 0.003*"like" + 0.003*"cloud" + '
  '0.003*"digital" + 0.002*"customer"')]

```
Определение доминирующей темы и её процентного вклада
```
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data_ready):
    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)
```

Самое представительное предложение для каждой темы
```
pd.options.display.max_colwidth = 100
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], axis=0)
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
sent_topics_sorteddf_mallet.head(10)

```
Частота распределения количества слов в документах
```
import re, numpy as np
doc_lens = [len(d) for d in df_dominant_topic.Text]
plt.figure(figsize=(10,4))
plt.hist(doc_lens, bins = 1000, color='navy')
plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))
plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,1000,9))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()
```
 ![image](https://github.com/cranberriess/thematic-modeling/assets/105839329/e3d16ea2-d7d0-4386-b86a-c04cf149ec87)<br>

Визуализация
```
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import gensim
!pip install "pandas<2.0.0"
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()
p = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
```
![image](https://github.com/cranberriess/thematic-modeling/assets/105839329/a463d2c4-9f38-4d09-9804-645ffcdc1b44)<br>
```

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
cloud = WordCloud(stopwords=stopwords_en,
                  background_color='white',
                  width=500,
                  height=400,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)
topics = lda_model.show_topics(formatted=False)
fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=75)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
```
![image](https://github.com/cranberriess/thematic-modeling/assets/105839329/bf05ab73-69e4-4e53-8d4e-5492090b34fa)<br>

## 4.	Вывод на основе графиков о полученной тематической модели, выделение тематики каждого кластера.
Мы получили оптимальное количество тем – 6, это темы: 
0 – объекты на изображениях, использование API (object, would, data, images, openai, use, api, also, able, using) 
1 – обучение модели, использование порождающих алгоритмов (data, ai, researchers, used, model, new, learning, generative, use, also) 
2 – классификация изображений (images, training, data, class, one, learning, algorithm, signal, group, classifier)  
3 – получение данных по ключам, кластеризация (redis, data, keys, ai, like, form, digital, cluster, code, user)
4 – исследование ИИ, конфиденциальность и точность (ai, data, researchers, learning, channels, privacy, accuracy, fsl, intelligence, power)
5 – генерация данных с использованием ИИ, новое обучение (ai, power, data, fashion, generative, new, learning, cost, majority, privacy)
По графику Intertopic Distance Map (via multidimensional scaling) можно заметить, что наиболее обширный кластер – первый, который включает в себя исследование данных ИИ (38% токенов), наименее – последний про исследование ИИ (3.9%). В облаке слов некоторые слова (data, ai) повторяются, так как все статьи затронули тему искусственного интеллекта и данных, также часто встречаются слова, относящиеся к теме обучения. Тем не менее тематики получились разные, несмотря на общую тему ИИ. 
