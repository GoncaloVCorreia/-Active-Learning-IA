# Spam Detection on YouTube Comment Section
The aim of this project is to create a machine learning model based on Active Learning to identify and filter out spam in YouTube comments. 

## Dependecies
1. Modal and sckit learn -> pip install modAL scikit-learn matplotlib -qqq  # matplotlib is optional
2. Rubrix framework -> pip install "rubrix[server]==0.18.0"
3. Install docker desktop

## How to run the code

1. docker run -d --name elasticsearch-for-rubrix -p 9200:9200 -p 9300:9300 -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2
2. python -m rubrix
3. Open a terminal in the source directory
4. python divide.py (not necessary because the divide data sets are already saved in the data directory)
5. python main.py
6. Annotate the comments in the rubrix link (http://0.0.0.0:6900 or http://localhost:6900) given (username: rubrix, password: 1234)
7. python train.py
