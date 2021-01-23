# hmm-viterbi-FB
A group project we did for BT3102: Computational Methods for Business Analytics, where we had to build, from scratch, 
implementations of the Viterbi and Forward-Backward algorithms using the frameworks of hidden Markov models.

We had two datasets: 
1. a list of tweets, where we took the Parts-of-Speech (POS) of each word/punctuation in the tweet as hidden states.
`twitter_train.txt` is pre-tagged (supervised) `twitter_train_no_tag.txt` while is not pre-tagged (unsupervised).
The full list of POS can be found in `twitter_tags.txt`. 
Using the Viterbi algorithm, we achieved a respectable tagging accuracy of 76%.


2. a list of sequences of price changes between successive transactions of Caterpillar stock, ranging from -6 cents to 6 cents.
The order of appearance of each sequence may not reflect real time, and sequences may not be consecutive (there might be gaps between consecutive sequences), 
while some sequences might contain subsequences of price changes that overlap with other sequences. 
Therefore, we modelled each sequence as a standalone Markov chain, independent of the other sequences in the list in our dataset.
`cat_price_changes_train.txt` is not pre-tagged (unsupervised). Because of the arbitrariness of the assignment of emissions to hidden states 
in the case of stock price changes, we have no supervised dataset for the Caterpillar's. Instead, we run the Forward-Backward algorithm to learn the 
output and transition probabilities, which we then feed into a modified form of the Viterbi algorithm to predict the most likely price change to occur
immediately following each sequence of price changes. Our Forward-Backward algorithm was trained with 3 hidden states which, when tested, appeared to 
approximate negative, positive and zero values respectively. Our predictions are in `cat_predictions.txt`

More details in our report, `BT3102_Project_Q6-7_report.pdf`.
