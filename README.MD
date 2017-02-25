
### 1) simple exploration

Simple exploratory notebook used to get a feel of the dataset
can be found here:  
https://github.com/i008/nyyelp/blob/master/exploration.ipynb

### 2) NLP project 

The notebook for this project can be found here:  
https://github.com/i008/nyyelp/blob/master/nlp.ipynb


#### main objective:
Predict review rating (how many stars the reviewer gave to given POI) given the textual-content of the review using pretrained **GLOVE** word embeddings and a 2 layers deep **GRU(LSTM)** recurrent neural network. Visualize clusters on a 2D space using PCA and t-SNE.

#### Steps(rough plan):

1) Process .text data from yelp review-documents
- filter to leave english-only reviews
- balance the dataset (use similar amount of data for each star rating (1,2,3,4,5*)
- balance the dataset by review length, so the length distribution of reviews for each rating-class is atleast similar (bucketing might be a good idea)

2) Process text for deep-learning
- bring  GLOVE embeddings to a reasonable form
- Tokenize each review 
- transform reviews into tokenized sequences
- Pad sequences to a fixed length
- Prepare the embedding weight-matrix to be used in the Embedding layer (we will not train this layer)
- split into test and train


3) Deep learning
- loss function, regression(rmse) and/or classification(logloss)
- prepare architecture (2-3 LSTM layers -> Dense)
- train, wish for the best

4) Post learning
- extract features from last LSTM layer
- casting to 2D space using t-SNE

5) Remarks
- we have to limit the amount of data, bc of memory issues possible solution is to save the sequences into HDF and flowing from disk during training

#### Summary

1) Data used:  
- For performance and memory reasons the final model was trained on a balanced and downsampled (by a factor of 10) subset  of the dataset Leaving us aorund 150,000 reviews (in total)

2) Deep Learning Model  
<img src="ims/model.png" width="500">

- The Embedding layer used GLOVE (http://nlp.stanford.edu/projects/glove/) weights and training of this layer was disabled
- The objective function was logloss wich means that it was a classification task , altough regression would also  be a valid option

3) Results 
 
confusion matrix:    
<img src="ims/cat_confusion_matrix.png" width="500">

classification report:   
<img src="ims/classification_report.png" width="500">










