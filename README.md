# Active-Learning-IA

## Dependecies
1. Modal and sckit learn -> pip install modAL scikit-learn matplotlib -qqq  # matplotlib is optional
2. Rubrix framework -> pip install "rubrix[server]==0.18.0"
3. Install docker desktop

## How to run the code

1. docker run -d --name elasticsearch-for-rubrix -p 9200:9200 -p 9300:9300 -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2
2. python -m rubrix
3. python main.py
4. Annotate the comments in the rubrix link given (username: rubrix, password: 1234)
5. python train.py
