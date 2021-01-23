# hmm-viterbi-FB
A group project we did for BT3102: Computational Methods for Business Analytics, where we had to build, from scratch, 
implementations of the Viterbi and Forward-Backward algorithms using the frameworks of hidden Markov models.

We had two datasets: 
1. a list of tweets, where we took the Parts-of-Speech (POS) of each word/punctuation in the tweet as hidden states.
`twitter_train.txt` is pre-tagged (supervised) `twitter_train_no_tag.txt` while is not pre-tagged (unsupervised).
The full list of POS can be found in `twitter_tags.txt`. 
Using the Viterbi algorithm, we achieved a respectable tagging accuracy of 76%.


2. a list of sequences of price changes between successive transactions of Caterpillar stock, ranging from -6 cents to 6 cents.
The order of appearance between sequences may not reflect real time - sequences may not be consecutive (there might be time gaps between adjacent sequences), 
while some sequences might overlap with others (i.e. contain subsequences of price changes that appear in other sequences 
due to collection of data from overlapping time frames).
Therefore, we modelled each sequence as a standalone Markov chain, independent of the other sequences in the list in our dataset.
`cat_price_changes_train.txt` is not pre-tagged (unsupervised). As compared to POS-tagging, the assignment of emissions to hidden states 
in the case of stock price changes is somewhat arbitrary. Obviously, then, we have no supervised version for this dataset.
Instead, we run the Forward-Backward algorithm to learn the output and transition probabilities, 
which we then feed into a modified form of the Viterbi algorithm to predict the most likely price change to occur
immediately following each sequence of price changes. Our Forward-Backward algorithm was trained with 3 hidden states which, when tested, appeared to 
approximate negative (-6 to -2), positive (2 to 6) and near-zero (-1 to 1) values respectively. 
The assignment pattern varies depending on the initialisation of the random seed.
In other initialisations that we tried, the hidden states seem to approximate extreme (-6,-5, 5, 6), middling (-4,-3,-2,2,3,4) and near-zero (-1,0,1) values respectively.
This added ambiguity is expected as it was left to the algorithm to learn the transition and output probabilities, with no supervision.
Our predictions are in `cat_predictions.txt`.

More details in our report, `BT3102_Project_Q6-7_report.pdf`.
