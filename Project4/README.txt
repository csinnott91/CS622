1. utilities.py

def generate_vocab(dir, min_count, max_files):

After initializing some variables, we first iterate through our positive and negative class folders and extract a set number of text 
files equal to our max_files argument. If this is -1, we just take ALL text files from these folders.

This is written to a massive string variable, which is then passed through sklearn's CountVectorizer function. This extracts each
unique word from the variable as well as the frequency with which is occurs. We then iterate across those frequencies and, provided
they exceed our min_count threshold, pass them to our vocab output.

def create_word_vector(fname, vocab):

Here, we create a function that takes a text file, and extracts the feature vector of that text file using our vocabulary (vocab)
created with the previous function. The lifting here is also done by sklearn's CountVectorizer.

def load_data(dir, vocab, max_files):

With this function we follow a similar function to our first. We iterate through max_files number of files in the positive and negative
class folders, then extract the feature vector of those text files. Those are subsequently appended to a numpy array. Labels are appended
to a separate array depending on if the text file was extraacted from the positive or negative class folder.

Note: Little to no filtering of things like punctuation, symbols, escape characters, etc. occurs in this. Some of this functionality seems
to be built into sklearn, but 15 hours later (and much fretting over intricacies of Windows vs. non-Windows file structures); I was just 
happy it was functioning "relatively" well even if it is incredibly ugly.

#####

2. ml.py

def MODEL_train():

I've condensed the various training functions (def dt_train(), def kmeans_train(), etc.) here as they all largely follow the same structure.
Using the relevant sklearn submodule, we first create a model object, then fit our feature vector array and label array to it. Some of these
have hyperparameters (e.g. hidden layer amount for the neural network training function), and these hyperparameters are passed appropriately
into arguments using sklearn's model object functions.

I ran into two issues during this. The first (less severe), is that sklearn's KMeans function does not perform well AT ALL on the data (unlike the output).
I'm not sure what is driving this, but I suspect it may be tied to my utilities module. The second, more severe issue is that the sklearn method for 
k-nearest neighbors crashes out with this data, throwing a ValueError. While I suspect this may be due to my utilities module as well, I have no idea
why it is throwing this error.

def model_test(X, model):

Here, we leverage sklearn's module syntax to our benefit. We pass our data and a model object into the function, then call the ".predict"
attribute from that model object.

def compute_F1(Y, Y_hat):

Again, we use sklearn's f1_score function to calculate the F1 score/value based off our test labels and our predicted labels for the data.

Note: At first blush the test script will not run past the KNN algorithm. Once that is commented out it runs through the remaining algorithms.