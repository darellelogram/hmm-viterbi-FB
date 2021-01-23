import math
import csv
import random
import re
import os
ddir = r'C:\\Users\darre\Documents\\NUS\BT3102\\Project\\projectfiles forwardbackwards edition\\projectfiles 131120 0103H\\projectfiles'
# Implement the six functions below
#### Qn 2a

XY = dict() # dict of number of times token x, tag y appear jointly in training data
Y = dict() # dict of number of times tag y appears in training data
uniqueX = set() # set of unique words that appear in training data

def qn2a(training_data_filename, output_probs_filename):        
    with open(training_data_filename, newline='', encoding="utf-8") as file:
        for line in file.readlines():
            line = line.rstrip()
            line = line.split()
            if len(line) == 2:
                w = line[0]
                j = line[1]
                
                if j not in Y.keys():
                    Y[j] = 1
                else:
                    Y[j] += 1

                if (w,j) not in XY.keys():
                    XY[(w,j)] = 1
                else:
                    XY[(w,j)]+= 1

                uniqueX.add(w)
    
    delta = 1/len(uniqueX)
    with open(output_probs_filename, 'w', encoding='utf-8') as output:
        for key, val in XY.items():
            w = key[0]
            j = key[1]
            prob = str((val + delta)/(Y[j] + 1 * (len(uniqueX) + delta)))
            output.write('\n' + w + '\t' + j + '\t' + prob)
    return Y

#### Qn 2b
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    
    Y = qn2a(f'{ddir}\\twitter_train.txt', f'{ddir}\\naive_output_probs.txt')

    with open(in_output_probs_filename, 'r', encoding='utf-8') as fin:
        probs = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
           
    # dictionary mapping tokens to tags y=j and prob P(x=w|y=j)
    # e.g. token: [{'tag': 'V', 'prob': 0.003}, ...]
    candidates = dict()

    for l in probs:
        line = l.split()
        if len(line) > 2:
            w,j,p  = line[0],line[1],line[2]
            if w not in candidates:
                candidates[w] = [{'tag': j, 'prob': p}]
            else:
                candidates[w].append({'tag': j, 'prob': p})
       
    # dictionary mapping tokens to most likely tag and its probability
    # e.g. token: {'tag': 'V', 'prob': 0.003}
    predictions = dict()
    
    for key, val in candidates.items():
        predictions[key] = sorted(val, key = lambda x: x['prob'], reverse=True)[0]

    with open(in_test_filename, 'r', encoding='utf-8') as fin:
        tokens = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
    with open(out_prediction_filename, 'w', encoding='utf-8') as fin:
        for token in tokens:
            if token in predictions:
                fin.write(predictions[token]['tag'] + '\n')
            else:
                # to maximise P(x = w|y=j) when w has not appeared in the test data,
                # we have to get the y=j with the smallest count
                tags_sorted = sorted(Y.items(), key = lambda x: x[1], reverse = False)
                fin.write(tags_sorted[0][0] + '\n')


#### Qn 3

# P(y = j|x = w) = P(y = j, x = w) / P(x = w)
# initialize dicts:
# X to store number of times word x=w appears
# Y to store number of times tag y=j appears
# XY to store number of times word x=w appears with tag y=j
X = dict()
Y = dict()
XY = dict()

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    with open(in_train_filename, newline='', encoding="utf-8") as file:
        data = csv.reader(file, delimiter='\t')
        
        # count total number of words in the data
        count = 0
        for line in data:
            if len(line) >= 2:
                w = line[0]
                j = line[1]
                
                if w not in X.keys():
                    X[w] = 1
                else:
                    X[w] += 1
                
                if j not in Y.keys(): 
                    Y[j] = 1 
                else: 
                    Y[j] += 1

                if (w,j) not in XY.keys():
                    XY[(w,j)] = 1
                else:
                    XY[(w,j)]+= 1
            count += 1
    
    # dictionary mapping tokens x=w to list of tags y=j with which it jointly occurs 
    # and the corresponding probs P(x=w|y=j) 
    # e.g. token: [{'tag': 'V', 'prob': 0.004}, {'tag': 'O', 'prob': 0.11}, ...]
    candidates = dict()

    with open(in_output_probs_filename, 'r', encoding='utf-8') as fin:
        probs = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

        for l in probs:
            line = l.split()
            if len(line) > 2:
                w,j,p  = line[0],line[1],line[2]

                j_count = 1
                if j in Y:
                    j_count = Y[j]
                # initialize to 1 if x=w has not appeared before
                w_count = 1
                if w in X:
                    w_count = X[w]
                # apply Bayes Rule to transform from P(y=j|x=w) to P(x=w|y=j)
                # + smoothing
                p = float(p) * (j_count/(float(count))) / (w_count/(float(count) + 1))
                
                if w not in candidates:
                    candidates[w] = [{'tag': j, 'prob': p}]
                else:
                    candidates[w].append({'tag': j, 'prob': p})
       
    # dictionary mapping tokens to most likely tag and its probability
    # e.g. token: {'tag': 'V', 'prob': 0.003}
    predictions = dict()
    
    for key, val in candidates.items():
        predictions[key] = sorted(val, key = lambda x: x['prob'], reverse=True)[0]

    with open(in_test_filename, 'r', encoding='utf-8') as fin:
        tokens = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
        
    # to maximise P(y=j|x=w) when w has not appeared in the test data,
    # we have to get the x=w with the smallest count
    leastFrequentX = sorted(X.items(), key = lambda x: x[1], reverse = False)[0][0]
    
    with open(out_prediction_filename, 'w', encoding='utf-8') as fin:
        for token in tokens:
            if token in predictions:
                fin.write(predictions[token]['tag'] + '\n')
            else:
                # when w has not appeared in the test data,
                # we write the x=w that appears least frequently
                fin.write(predictions[leastFrequentX]['tag'] + '\n')

def transition_probabilities(in_train_filename, out_prediction_filename):
    #dictionary of transition probabilities, key is (word t-1, word t)
    aij = dict()
    #dictionary of number of times we are in state Y at time t
    count_Y_t = dict()
    START = '*'
    STOP = '.'

    storage = []

    with open(in_train_filename, newline='', encoding="utf-8") as file:
        csvfile = csv.reader(file, delimiter='\t')    
        for rows in csvfile:
            storage.append(rows)
        
    y_t, y_t_1 = '', ''
    for i in range (0, len(storage)):
        if i == 0 or len(storage[i-1]) == 0:
            y_t_1 = START
            y_t = storage[i][1]
            
            a = (y_t_1,y_t)
            if a not in aij.keys():
                    aij[a] = 1
            else:
                aij[a] += 1

            if y_t_1 not in count_Y_t.keys():
                count_Y_t[y_t_1] = 1
            else:
                count_Y_t[y_t_1] += 1

        elif len(storage[i]) == 0:
            y_t_1 = storage[i-1][1]
            y_t = STOP
            a = (y_t_1,y_t)
            if a not in aij.keys():
                    aij[a] = 1
            else:
                aij[a] += 1

            if y_t_1 not in count_Y_t.keys():
                count_Y_t[y_t_1] = 1
            else:
                count_Y_t[y_t_1] += 1
               
        else:
            y_t = storage[i][1]
            y_t_1 = storage[i-1][1]
            a = (y_t_1,y_t)  
            if a not in aij.keys():
                    aij[a] = 1
            else:
                aij[a] += 1

            if y_t_1 not in count_Y_t.keys():
                count_Y_t[y_t_1] = 1
            else:
                count_Y_t[y_t_1] += 1                

    with open(out_prediction_filename, 'w', encoding='utf-8') as fin:
        for key, val in aij.items():
            b = val/count_Y_t[key[0]] # where key = (i, j)
            fin.write(key[0] + '\t' + key[1] + '\t' + str(b) + '\n')

# transition_probabilities(f'{ddir}\\twitter_train.txt', f'{ddir}\\trans_probs.txt')

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    START = '*'
    STOP = '.'

    all_tags = [] # all possible tags that Y can take, excluding START and STOP
    with open(in_tags_filename, newline = '', encoding="utf-8") as file:
        lines = file.readlines()
        probs = []
        all_tags = []
        for l in lines:
            all_tags.append(l.rstrip())

    N = len(all_tags) # number of unique tags, excluding START and STOP

    # read in from in_trans_prob_filename
    # aij stores transition probabilities in dict of dicts
    # bv is dictionary of output probabilities with key being tuple (word, tag) with value as prob
    aij = dict()
    bv = dict()
    tweets_list = list()
    finalBPlist = list()
    with open(in_trans_probs_filename, newline= '', encoding="utf-8") as file:
        csvfile = csv.reader(file, delimiter='\t')
        for row in csvfile:
            if len(row) == 3:
                i = row[0]
                j = row[1]
                prob = float(row[2])
                if i not in aij.keys():
                    aij[i] = {j: prob}
                else:
                    aij[i][j] = prob
    
    # Populating dictionary bv from output_probs.txt
    with open(in_output_probs_filename, newline='', encoding="utf-8") as file2:
        csvfile2 = csv.reader(file2, delimiter='\t')
        for row in csvfile2:
            if len(row) == 3:
                word = row[0]
                tag = row[1]
                prob = float(row[2])
                key = (word, tag)    
                bv[key] = prob 
    
    #Make list of tweets (list of lists)
    with open(in_test_filename, encoding = "utf8") as test:
        lines = test.readlines()
        curr_tweet = []
        for rows in lines:
            parameters = rows.rstrip()
            if len(parameters) != 0:
                curr_tweet.append(parameters)
            elif len(curr_tweet) != 0 and len(parameters) == 0:
                tweets_list.append(curr_tweet)
                curr_tweet = []
            else:
                continue

    with open(out_predictions_filename, 'w', encoding='utf-8') as file:
        #Performing viterbi on each tweet in our list of tweets
        for tweet in tweets_list: #tweet is a list of words
            #START VITERBI HERE
            n = len(tweet) # number of tweets
            likeliest_probs = dict() #  big Pi
            BP = dict() #backpointers
            
            likeliest_probs[0] = dict()
            BP[0] = dict()
            delta = 1/len(uniqueX)
            for v in aij.keys(): # for all states
                # index 0 represents step t = 1
                if v in aij[START].keys(): # if train data contains transition START -> v
                    possible_key = (tweet[0],v)
                    if possible_key in bv.keys():
                        prob = aij[START][v] * bv[possible_key]
                        likeliest_probs[0][v] = prob
                    else:
                        prob = aij[START][v] * (delta) /(delta + len(X)) #smoothing
                        likeliest_probs[0][v] = prob              
                else:
                    aij[START][v] = 1/(1+N**2)
                    possible_key = (tweet[0],v)
                    if possible_key in bv.keys():
                        prob = aij[START][v] * bv[possible_key]
                        likeliest_probs[0][v] = prob
                    else:
                        prob = aij[START][v] * (delta) /(delta + len(X)) #smoothing
                        likeliest_probs[0][v] = prob
                BP[0][v] = []                  

            for k in range(1, n):
                likeliest_probs[k] = dict()
                BP[k] = dict()
                for v in aij.keys(): # for all states
                    likeliest_state = ''
                    highest_prob = -1
                    for u in likeliest_probs[k-1].keys():
                        new_prob = likeliest_probs[k-1][u]                      
                        if v in aij[u].keys():
                            if (tweet[k],v) in bv.keys():
                                new_prob = new_prob * aij[u][v] * bv[(tweet[k],v)]
                            else:
                                # smooth for bv - output probs
                                smooth = (delta) / (delta + len(X))
                                new_prob = new_prob * aij[u][v] * smooth
                            if new_prob > highest_prob:
                                likeliest_state = u
                                highest_prob = new_prob

                    likeliest_probs[k][v] = highest_prob
                    
                    listforBP = BP[k-1][likeliest_state].copy()
                    listforBP.append(likeliest_state)
                    BP[k][v] = listforBP
                    
            maxProb = 0
            finalBPlist_curr = []
            end_state = ''
        
            for v in aij.keys():
                if v in likeliest_probs[n-1].keys() and STOP in aij[v].keys():
                    prob = likeliest_probs[n-1][v] * aij[v][STOP]
                    if prob > maxProb:
                        maxProb = prob
                        finalBPlist_curr = BP[n-1][v]
                        end_state = v
            finalBPlist.extend(finalBPlist_curr)
            finalBPlist.append(end_state)
            finalBPlist.append('\n')
        for state in finalBPlist:
            file.write(state + '\n') 

        file.write('\n')
        
                
# Question 5
# How can you improve your POS tagger futher?

# We have 3 possible ways of improving our POS tagging further.

# The first method would be to better handle unseen words through utilizing a better form of smoothing. 
# Instead of randomly selecting a value we used a delta inversely proportional to the number of words in our corpus.

# For the second method, we will factor in the next word in addition to how we already account for the previous word.
# Words can be better inferred by both inferring the words in front of it and to the back of it.
# For example, "punch" can be both a noun and verb depending on the words that come before it, "fruit punch" (noun)
# while "punch him" (verb)

# For the third method, we will take advantage of linguistic patterns in the tweets
# We are able to identify certain patterns that we can tag the words to
# we can tag as, noun, adjective, verb and adverb based off the prefix and suffix
# we use Regular Expressions to identify this

def assign_to_word(word):
    if word.endswith(("age","ance","ant","ee","ence","ery","ess","ion","er","ist","ment","nss","or")):
         #tag as noun
         return "N"
    elif word.endswith(("able","ible","ful","ic","ian","less","ous")):
        #tag as adjective
        return "A"
    elif word.endswith(("ate","ify","ise","ize")):
        #tag as verb
        return "V"
    elif word.endswith(("ward","wards","wise","ly")):
        #tag as adverb
        return "R"
    elif word.startswith(("@USER_")):
        #tag as user
        return "@"
    elif word.startswith(("#")):
        #tag as hashtag
        return "#"
    elif word.startswith(("http")):
        #tag as URL
        return "U"
    elif word.startswith((".",",","?","!",":","(",")")):
        #tag as punctuation
        return ","
    elif word == "and" or word == "&" or word == "but" or word == "or":
        #tag as conjunction
        return "&"
        
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                      out_predictions_filename):
    viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                      out_predictions_filename)
    with open(out_predictions_filename, 'r', encoding = 'utf-8') as fin1:
        predicted_tags = [l.strip() for l in fin1.readlines() if len(l.strip()) != 0]

    with open(in_test_filename, 'r', encoding = 'utf-8') as fin2:
        words = [l.strip() for l in fin2.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(words)
    
    new_predicted_tags = []
    for word, tag in zip(words, predicted_tags):
        new_tag = assign_to_word(word)
        if new_tag:
            tag = new_tag
        new_predicted_tags.append(tag)

    with open(out_predictions_filename, 'w', encoding='utf-8') as file:
        for tag in new_predicted_tags:
            file.write(tag + '\n')
    

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    """
    # Step 1) read in total_state_list, tweet_list
    # Step 2) create aij and bv dictionaries
    # Step 3) initialise random variable for each aij, bv
    #  and normalise
    """
    # define constants
    START = '*'
    STOP = '.'
    random.seed(seed)

    ### HELPER FUNCTIONS ### 
    # take in a list of float values and out with normalized values
    def normalizeList(input_list): 
        normalized_list = []
        sum_of_list = sum(input_list)
        for num in input_list:
            output_num = num/sum_of_list
            # round to 5 d.p.
            rounded_num = round(output_num, 5 - int(math.floor(math.log10(abs(output_num))) - 1))
            normalized_list.append(rounded_num)
        return normalized_list
    
    def forward(t, j, word):
        if t == 1:
            return aij[(START, j)]*bv[(word, j)]
        # termination step
        elif j == STOP: # no output probabilities for STOP state
            sum = 0
            for i in all_tags:
                sum += aij[(i,j)] * alpha[t-1][i]
            return sum
        
        else:
            sum = 0
            for i in all_tags: # changed from total_state_list
                sum += aij[(i,j)]*alpha[t-1][i]
            return sum * bv[(word, j)]

    def backward(t, i, word, tweet_len):
        if t == tweet_len:
            return aij[(i,STOP)]

        else:
            sum = 0
            for j in all_tags:
                sum += aij[(i,j)]*beta[t+1][j] * bv[(word),j]
            return sum

    ### START OF FORWARD BACKWARD CODE ###
    # Step 1)
    # tweet_list is list of current tweets (list of strings) -> function operates on each tweet
    tweet_list = list()

    with open(in_train_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        curr_tweet = []
        for rows in lines:
            parameters = rows.rstrip() 
            if len(parameters) != 0:
                curr_tweet.append(parameters)
            elif len(curr_tweet) != 0 and len(parameters) == 0: # end of tweet
                tweet_list.append(curr_tweet)
                curr_tweet = []
            else:
                continue
    
    # Step 2)
    # Random initialization for transition and output probabilities
    aij = dict() # transition probabilities
    bv = dict() # output probabilities
    all_tags = [] # all possible tags that Y can take
    
    with open(in_tag_filename, newline = '', encoding="utf-8") as file:
        lines = file.readlines()
        # list of lists to store randomly generated probabilities
        probs = []
        all_tags = []
        for l in lines:
            all_tags.append(l.rstrip())

        # populate list of lists with random numbers
        for i in range(len(all_tags) + 2): # +2 to account for START and STOP states
            probs_over_all_j = []
            # each inner list should contain len(all_tags) number of random numbers 
            # to generate that many transition probabilities
            for j in range(len(all_tags)):
                probs_over_all_j.append(random.random())
            # normalize probabilities so that each inner list has total probs that add up to 1
            # because summing P(yt+1=j|yt=i) over all j should equal 1 
            probs.append(normalizeList(probs_over_all_j))

        # Now we initialize transition probs by retrieving the random numbers we generated earlier
        # initialize transition probs from START to all states except START and STOP
        for j in range(len(all_tags)):
            prob = probs[0][j]
            aij[(START, all_tags[j])] = prob

        # initialize transition probs between all states (excluding START and STOP)
        for i in range(1, len(all_tags)+1):
            for j in range(len(all_tags)):
                prob = probs[i][j]
                aij[(all_tags[i-1], all_tags[j])] = prob
        
        # initialize transition probs from all states (except START and STOP) to STOP
        for i in range(len(all_tags)):
            prob = probs[len(all_tags)+1][i]
            aij[(all_tags[i], STOP)] = prob

    # number of unique tags - excluding START and STOP
    N = len(all_tags) 

    # dict to store all occurrences of each unique word
    # these occurrences contain the params needed by forward() and backward()
    unique_words = dict()
    for tweet_num in range(len(tweet_list)):
        for t in range(len(tweet_list[tweet_num])):
            word = tweet_list[tweet_num][t]
            if word not in unique_words.keys():
                unique_words[word] = []
                # store the tweet_num and timestep where the word occurred
                unique_words[word].append((tweet_num,t))
            else:
                unique_words[word].append((tweet_num, t))
    
    # list of lists to store randomly generated probabilities
    probs = []
    for tag in all_tags:
        probs_over_all_words = []
        for word in unique_words.keys():
            probs_over_all_words.append(random.random())
        # normalize probabilities across all words emitted by a particular tag
        # because summing P(xt=word|yt=tag) over all words should equal 1
        probs.append(normalizeList(probs_over_all_words))
    for y in range(len(all_tags)):
        # initialize index of word to retrieve random number from probs list
        x = 0
        for word in unique_words.keys():
            bv[(word, all_tags[y])] = probs[y][x]
            x += 1


    # Qn 6 part (b)
    # Save output probabilities and transition probabilities right after initialization
    # into output_probs3.txt and trans_probs3.txt
    # then the run() fn will use these to run viterbi_predict() and 
    # save the predicted tags into fb_predictions3.txt
    trans_probs_filename3 =  f'{ddir}\\trans_probs3.txt'
    output_probs_filename3 = f'{ddir}\\output_probs3.txt'
    with open(trans_probs_filename3, 'w', encoding='utf-8') as file:
        for transition, prob in aij.items():
            file.write(str(transition[0]) + '\t' + str(transition[1]) + '\t' + str(prob) + '\n')
    with open(output_probs_filename3, 'w', encoding='utf-8') as file:
        for (word, tag), prob in bv.items():
            file.write(word + '\t' + tag + '\t' + str(prob) + '\n')

    # triple-nested dict to store epsilon values, keys for each level are:
    # 1) tweet number, 2) index of word in tweet, 3) i, j (evidence for being in state in i and j)
    epsilon = {}

    # triple-nested dict to store gamma values, keys for each level are:
    # 1) tweet number, 2) index of word in tweet, 3) j (being in state j given all other outputs)
    gamma = {}

    # store sum of log likelihood from previous iteration
    prev_sum_log_likelihood = 0
    counter = 0
    for num in range(max_iter): # iteration to repeate forward_backward algo
        counter += 1
        sum_log_likelihood = 0

        for tweet_num in range(len(tweet_list)):
            # Alpha is prob of outputting x1, x2, ... xt and being at state j at step t
            # Doubly-nested dict to cache alpha values
            # 1st level key is index t of word in tweet, 2nd level key is state j, value is the probability
            alpha = {}

            # Beta is prob of outputting future outputs xt, ... xn given that we are at current state yt = j
            # Doubly-nested dict beta to cache beta values
            # 1st level key is index t of word in tweet, 2nd level key is state j, value is the probability
            beta = {}

            tweet = tweet_list[tweet_num] #choosing the tweet to work on 

            for word_num in range(1, len(tweet)+1): # running through each word index
                alpha_t = word_num 
                beta_t = len(tweet) - word_num +1  # to find timestep of beta_t, need to -1
                alpha[alpha_t] = {} # initialise dict for key of current timestep t
                beta[beta_t] = {} # initialise dict for key of current timestep t
                for j in all_tags: # get alpha, beta for all tags
                    alpha[alpha_t][j] = forward(alpha_t, j, tweet[alpha_t -1])
                    # if beta_t timestep is at the end of the tweet, beta = 1, since there is no prev word to look at
                    if (beta_t == len(tweet)):
                        beta[beta_t][j] = backward(beta_t, j, "", len(tweet))
                    else:
                        beta[beta_t][j] = backward(beta_t, j, tweet[beta_t], len(tweet))
            # Termination 
            alpha[len(tweet)+1] = {}
            alpha[len(tweet) +1][STOP] = forward(len(tweet)+1, STOP,"")
            beta[0] = {}
            beta[0][START] = backward(0,START,tweet[0], len(tweet))

            # handle time step t=0 for alpha and time step t=n+1 for beta
            # since there are no past timesteps before y0 = START, 
            # the probability of observing any sequence of past states given state START is 1
            alpha[0] = {START:1}
            # since there are no more timesteps after yn+1 = STOP,
            # probability of observing any future states given state STOP is 1
            beta[len(tweet) +1] = {STOP:1}
            
            # Start of E-step: Find gamma and epsilon
            denom_gamma = {}
            denom_epsilon = alpha[len(tweet) +1][STOP]
            
            # cache denominators for gamma values
            for word_num in range(1, len(tweet) +1):
                # denominator of gamma is unique for each index t of word in tweet
                # but not unique across each state j
                denom_gamma[word_num] = 0
                for j in all_tags:
                    denom_gamma[word_num] += alpha[word_num][j] * beta[word_num][j]
            
            # Triply-nested dictionary gamma, keys for each level are:
            # 1) tweet number, 2) index t of word in tweet, 3) state j
            # value is the gamma value
            gamma[tweet_num] = {}

            # Triply-nested dictionary epsilon, keys for each level are:
            # 1) tweet number, 2) index t of word in tweet, 3) tuple of states (i,j)
            # value is the epsilon value
            epsilon[tweet_num] = {}

            # for gamma
            for word_num in range(1, len(tweet) +1):
                gamma[tweet_num][word_num] = {}
                for j in all_tags:
                     gamma_numer = (alpha[word_num][j] * beta[word_num][j])
                     gamma[tweet_num][word_num][j] = gamma_numer / denom_gamma[word_num]
            
            # for epsilon
            for word_num in range(0, len(tweet)+1): # 0 to n where n is len(tweet) - Also accounts for start and stop
                epsilon[tweet_num][word_num] = {}
                for j in all_tags:
                    # handle special case at start of tweet
                    if word_num == 0:
                        epsilon_numer = alpha[word_num][START] * aij[(START,j)] * bv[(tweet[word_num],j)] * beta[word_num+1][j]
                        epsilon[tweet_num][word_num][(START,j)] = epsilon_numer/denom_epsilon
                        continue
                    # handle special case at end of tweet
                    elif word_num == len(tweet):
                        # beta[word_num+1][STOP] is handled upstairs in line 656
                        epsilon_numer = alpha[word_num][j] * aij[(j,STOP)] * beta[word_num+1][STOP]
                        epsilon[tweet_num][word_num][(j,STOP)] = epsilon_numer/denom_epsilon
                        continue

                    for i in all_tags:
                        epsilon_numer = alpha[word_num][i] * aij[(i,j)] * bv[(tweet[word_num], j)] * beta[word_num+1][j] 
                        epsilon[tweet_num][word_num][(i,j)] = epsilon_numer/denom_epsilon
            
            # likelihood of observing all data in the tweet from x1, x2... xn
            # we use the alpha value of being in state STOP at t=n+1
            likelihood = alpha[len(tweet) +1][STOP]
            log_likelihood = math.log(likelihood)
            sum_log_likelihood += log_likelihood

        # Start of M-step
        # Updated transition probabilities aij and output probabilities bv

        # initialize trans probs from START to all other states
        denom_aSTARTj = 0
        for j in all_tags:
            for tweet_num in range(len(tweet_list)):
                denom_aSTARTj += epsilon[tweet_num][0][(START,j)]
        for j in all_tags:
            numer_aSTARTj = 0
            for tweet_num in range(len(tweet_list)):
                numer_aSTARTj += epsilon[tweet_num][0][(START,j)]
            aij[(START, j)] = numer_aSTARTj/denom_aSTARTj
                        
        # Update transition probabilities
        for i in all_tags:
            denom_aij = 0
            for tweet_num in range(len(tweet_list)):
                tweet = tweet_list[tweet_num]
                for word_num in range(1, len(tweet)+1):
                    if word_num == len(tweet):
                        denom_aij += epsilon[tweet_num][len(tweet)][(i, STOP)]
                        continue
                    for j in all_tags:
                        denom_aij += epsilon[tweet_num][word_num][(i,j)]

            for j in all_tags:
                numer_aij = 0
                for tweet_num in range(len(tweet_list)):
                    tweet = tweet_list[tweet_num]
                    for word_num in range(1, len(tweet)):
                        numer_aij += epsilon[tweet_num][word_num][(i,j)]
                aij[(i,j)] = numer_aij/denom_aij
            
            numer_aiStop = 0
            for tweet_num in range(len(tweet_list)):
                tweet = tweet_list[tweet_num]
                numer_aiStop += epsilon[tweet_num][len(tweet)][(i,STOP)]
            aij[(i,STOP)] = numer_aiStop/denom_aij
                        
        # update bv
        # find denom first
        for j in all_tags:
            denom_bv = 0
            for tweet_num in range(len(tweet_list)):
                tweet = tweet_list[tweet_num]
                for word_num in range(1,len(tweet)+1):
                    denom_bv += gamma[tweet_num][word_num][j]

            for word, occurrences in unique_words.items():
                numer_bv = 0
                for tup in occurrences:
                    tweet_num = tup[0]
                    t = tup[1] # retrieve timestep t at which word occurs
                    numer_bv += gamma[tweet_num][t+1][j]
                bv[(word,j)] = numer_bv/denom_bv
        
        diff_likelihood = abs(sum_log_likelihood - prev_sum_log_likelihood)
        if abs(sum_log_likelihood - prev_sum_log_likelihood) < thresh:
            break
        prev_sum_log_likelihood = sum_log_likelihood
        print(diff_likelihood)

    with open(out_trans_filename, 'w', encoding='utf-8') as file:
        for key, prob in aij.items():
            file.write(str(key[0]) + '\t' + str(key[1]) + '\t' + str(prob) + '\n')
    
    with open(out_output_filename, 'w', encoding='utf-8') as file:
        for key, prob in bv.items():
            file.write(str(key[0]) + '\t' + str(key[1]) + '\t' + str(prob) + '\n')

def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_filename):

    START = '*'
    STOP = '.'

    all_tags = [] # all possible tags that Y can take
    in_tags_filename       = f'{ddir}\\cat_states.txt'
    with open(in_tags_filename, newline = '', encoding="utf-8") as file:
        lines = file.readlines()
        probs = []
        all_tags = []
        for l in lines:
            all_tags.append(l.rstrip())

    N = len(all_tags) # number of unique tags, excluding START and STOP

    # read in from in_trans_prob_filename
    # store transition probabilities in dict of dicts aij
    # bv is dictionary with key being tuple (word, tag) which leads to prob
    aij = dict()
    bv = dict()
    tweets_list = list()
    finalBPlist = list()
    with open(in_trans_probs_filename, newline= '', encoding="utf-8") as file:
        csvfile = csv.reader(file, delimiter='\t')
        for row in csvfile:
            if len(row) == 3:
                i = row[0]
                j = row[1]
                prob = float(row[2])
                if i not in aij.keys():
                    aij[i] = {j: prob}
                else:
                    aij[i][j] = prob
    
    uniqueX = set()
    # Populating dictionary from  output_probs.txt
    with open(in_output_probs_filename, newline='', encoding="utf-8") as file2:
        csvfile2 = csv.reader(file2, delimiter='\t')
        for row in csvfile2:
            if len(row) == 3:
                word = row[0]
                tag = row[1]
                prob = float(row[2])
                if tag not in bv.keys():
                    bv[tag] = dict()
                bv[tag][word] = prob
                if word not in uniqueX:
                    uniqueX.add(word)

    # delta is the smoothing constant
    delta = 1/len(uniqueX)
    
    #Make list of tweets (list of lists)
    with open(in_test_filename, encoding = "utf8") as test:
        lines = test.readlines()
        curr_tweet = []
        for rows in lines:
            parameters = rows.rstrip()
            if len(parameters) != 0:
                curr_tweet.append(parameters)
            elif len(curr_tweet) != 0 and len(parameters) == 0:
                tweets_list.append(curr_tweet)
                curr_tweet = []
            else:
                continue

    #Performing viterbi on each tweet in our list of tweets
    last_state_list = []
    for tweet in tweets_list: #tweet is a list of words
        #START VITERBI HERE
        n = len(tweet) # number of tweets
        likeliest_probs = dict() #  this is big Pi
        BP = dict() #backpointers
        
        likeliest_probs[0] = dict()
        BP[0] = dict()
        for v in aij.keys(): # for all states - range(1, N):
            # index 0 represents step t = 1
            if v in aij[START].keys(): # if train data contains transition START -> v
                possible_key = (tweet[0],v)
                if possible_key in bv.keys():
                    prob = aij[START][v] * bv[possible_key]
                    likeliest_probs[0][v] = prob
                else:
                    prob = aij[START][v] * delta #smoothing
                    likeliest_probs[0][v] = prob              
            else:
                aij[START][v] = 1/(1+N**2)
                possible_key = (tweet[0],v)
                if possible_key in bv.keys():
                    prob = aij[START][v] * bv[possible_key]
                    likeliest_probs[0][v] = prob
                else:
                    prob = aij[START][v] * delta #smoothing
                    likeliest_probs[0][v] = prob
            BP[0][v] = []                  

        for k in range(1, n):
            likeliest_probs[k] = dict()
            BP[k] = dict()
            for v in aij.keys(): # for v = 1 to N-1
                likeliest_state = ''
                highest_prob = -1
                for u in likeliest_probs[k-1].keys():
                    new_prob = likeliest_probs[k-1][u]                      
                    if v in aij[u].keys():
                        if (tweet[k],v) in bv.keys():
                            new_prob = new_prob * aij[u][v] * bv[(tweet[k],v)]
                        else:
                            # smooth for bv - output probs
                            smooth = delta
                            new_prob = new_prob * aij[u][v] * smooth
                        if new_prob > highest_prob:
                            likeliest_state = u
                            highest_prob = new_prob

                likeliest_probs[k][v] = highest_prob
                
                listforBP = BP[k-1][likeliest_state].copy()
                listforBP.append(likeliest_state)
                BP[k][v] = listforBP
                
        maxProb = 0
        finalBPlist_curr = []
        end_state = ''
    
        for v in aij.keys():
            if v in likeliest_probs[n-1].keys() and STOP in aij[v].keys():
                prob = likeliest_probs[n-1][v] * aij[v][STOP]
                if prob > maxProb:
                    maxProb = prob
                    finalBPlist_curr = BP[n-1][v]
                    end_state = v
        finalBPlist.extend(finalBPlist_curr)
        finalBPlist.append(end_state)
        finalBPlist.append('\n')
        last_state_list.append(end_state)
  
    total_dict = {}
    for i in all_tags:
        total_dict[i] = {}
        total = 0
        for j in all_tags:
            total += aij[i][j]
            total_dict[i][j] = total
            
    final_states_list = []
    for tag in last_state_list:
        random_number = random.uniform(0, total_dict[tag][all_tags[-1]])
        dictionary = total_dict[tag]
        for key in all_tags:
            number = dictionary[key]
            random_number -= number
            if random_number <= 0:
                final_states_list.append(key)
                break

    final_emission_list = []
    for state in final_states_list:
        random_number = random.uniform(0,1)
        for emission in uniqueX:
            number = bv[state][emission]
            random_number -= number
            if random_number <= 0:
                final_emission_list.append(emission)
                break

    with open(out_predictions_filename, 'w', encoding='utf-8') as file:
        for emission in final_emission_list:
            file.write(emission + '\n')


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth))**2
    return error/len(predicted_tags), error, len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''
    
    ddir = r'C:\\Users\darre\Documents\\NUS\BT3102\\Project\\projectfiles forwardbackwards edition\\projectfiles 131120 0103H\\projectfiles'    
    in_train_filename = f'{ddir}\\twitter_train.txt'

    naive_output_probs_filename = f'{ddir}\\naive_output_probs.txt'

    in_test_filename = f'{ddir}\\twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}\\twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}\\naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}\\naive_predictions2.txt'
    #Y = qn2a(in_train_filename, naive_output_probs_filename)
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}\\trans_probs.txt'
    output_probs_filename = f'{ddir}\\output_probs.txt'

    in_tags_filename = f'{ddir}\\twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}\\viterbi_predictions.txt'
    # generate transition probabilities
    transition_probabilities(in_train_filename, trans_probs_filename)
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}\\trans_probs2.txt'
    output_probs_filename2 = f'{ddir}\\output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}\\viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename, viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}\\twitter_train_no_tag.txt'
    in_tag_filename     = f'{ddir}\\twitter_tags2.txt'
    out_trans_filename  = f'{ddir}\\trans_probs4.txt'
    out_output_filename = f'{ddir}\\output_probs4.txt'
    max_iter = 10 
    seed     = 777
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    trans_probs_filename3 =  f'{ddir}\\trans_probs3.txt'
    output_probs_filename3 = f'{ddir}\\output_probs3.txt'
    viterbi_predictions_filename3 = f'{ddir}\\viterbi_predictions3.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
                     viterbi_predictions_filename3)
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename4 =  f'{ddir}\\trans_probs4.txt'
    output_probs_filename4 = f'{ddir}\\output_probs4.txt'
    viterbi_predictions_filename4 = f'{ddir}\\viterbi_predictions4.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
                     viterbi_predictions_filename4)
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}\\cat_price_changes_train.txt'
    in_tag_filename     = f'{ddir}\\cat_states.txt'
    out_trans_filename  = f'{ddir}\\cat_trans_probs.txt'
    out_output_filename = f'{ddir}\\cat_output_probs.txt'
    max_iter = 1000000
    seed     = 777
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    in_test_filename         = f'{ddir}\\cat_price_changes_dev.txt'
    in_trans_probs_filename  = f'{ddir}\\cat_trans_probs.txt'
    in_output_probs_filename = f'{ddir}\\cat_output_probs.txt'
    in_states_filename       = f'{ddir}\\cat_states.txt'
    predictions_filename     = f'{ddir}\\cat_predictions.txt'
    cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
               predictions_filename)

    in_ans_filename     = f'{ddir}\\cat_price_changes_dev_ans.txt'
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    run()
