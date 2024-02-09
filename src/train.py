from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import rubrix as rb
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import csv
#Estatisticas
def stats(predictions, validations):
    tn=0
    fn=0
    tp=0
    fp=0
    for i in range(len(validations)):
        #tp
        if(predictions[i]==1 and int(validations[i])==1 ):
            tp+=1
        #tn
        if(predictions[i]==0 and int(validations[i])==0):
            tn+=1
        #fp
        if(predictions[i]==1 and int(validations[i])==0 ):
            fp+=1
        #fn
        if(predictions[i]==0 and int(validations[i])==1 ):
            fn+=1
            
    #print(tn)
    #print(fn)
    #print(tp)
    #print(fp)
    
    sensetivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    erroRate=(fp+fn)/(tp+fp+tn+fn)
    precision=tp/(tp+fp)
    recall=tn/(tn+fp)
    f1score=(2*recall*precision)/(recall+precision)
    
    print("Sensetivity: "+str(sensetivity))
    print("Specificty: "+str(specificity))
    print("Error Rate: "+str(erroRate))
    print("Precision: "+str(precision))
    print("F1 score: "+str(f1score))

def train():
    train_df = pd.read_csv("data/PSY.csv")
    test_df = pd.read_csv("data/PSY.csv")
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Define our classification model
    #classifier = MultinomialNB()
    classifier =  MultinomialNB()

    # Define active learner
    learner = ActiveLearner(
        estimator=classifier,
        query_strategy=entropy_sampling
    )
    # The resulting matrices will have the shape of (`nr of examples`, `nr of word n-grams`)
    vectorizer = CountVectorizer(ngram_range=(1,5))# The resulting matrices will have the shape of (`nr of examples`, `nr of word n-grams`)


    X_train = vectorizer.fit_transform(train_df.CONTENT)
    X_test = vectorizer.transform(test_df.CONTENT)

    # Number of instances we want to annotate per iteration
    n_instances = 10

    # Accuracies after each iteration to keep track of our improvement
    accuracies = []
    x=0
    
    
    for x in range(1):
        query_idx, query_inst = learner.query(X_train, n_instances=n_instances)

        # get predictions for the queried examples
        try:
            print("entrei aqui")
            probabilities = learner.predict_proba(X_train[query_idx])
            print("--------")
            
        # For the very first query we do not have any predictions
        except NotFittedError:
            print("entrei aqui2")
            probabilities = [[0.5, 0.5]]*n_instances
            print("--------")
                

        records = [
            rb.TextClassificationRecord(
                id=idx,
                text=train_df.CONTENT.iloc[idx],
                prediction=list(zip(["HAM", "SPAM"], probs)),
                prediction_agent="MultinomialNB",
            )
            for idx, probs in zip(query_idx, probabilities)

        ]
        if(x!=0):
            rb.log(records, name="active_learning_tutorial")
            input("Press Enter to continue...")


        # Load the annotated records into a pandas DataFrame
        records_df = rb.load("active_learning_tutorial", ids=query_idx.tolist()).to_pandas()

        # check if all examples were annotated
        print(records_df)
        print(records_df.annotation)
        if any(records_df.annotation.isna()):
            raise UserWarning("Please annotate first all your samples before teaching the model")
        x=0
        z=0
        while(z<5):
            # train the classifier with the newly annotated examples
            y_train = records_df.annotation.map(lambda x: int(x == "SPAM"))
            learner.teach(X=X_train[query_idx], y=y_train.to_list())
            z+=1
            accuracies.append(learner.score(X=X_test, y=test_df.CLASS))
          
            # Keep track of our improvement
            
            x+=1

            print("Accuracy: "+str(learner.score(X=X_test, y=test_df.CLASS)))

            test_df2=open("data/PSY.csv","r")
            ##VALiDATION
            reader=csv.reader(test_df2)
            data=list(reader)
            validate=[]
            for c in data:
                validate.append(c[4])
            validate.pop(0)

            
            ##PREDICTOIN
            predictions=learner.predict(X_test)
            
            #Calcula estatistica
            stats(predictions,validate)
    
    #plots
    print(accuracies)
    plt.plot(accuracies)
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy");
    plt.show()

if __name__ == "__main__":
    train()
    

