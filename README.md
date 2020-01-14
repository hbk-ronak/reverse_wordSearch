# Reverse Word Search

### Inspiration

Often times, when writing an essay, we are at loss of words to describe something or we are too verbose. It would be a fun application if someone could enter a sentence or a phrase and the application could return top five word suggestions

### Implementation

1. Scrape the word definition from an online source
2. Using Word embeddings find a vector for the definition
3. Cluster the words based on the vectors
4. Take input from the user
5. Find the best match cluster
6. Find the best match words

### Improvements

1. Use a larger dictionary (probably Webster's)
2. If larger dictionary available train a doc2vec model on the definition
3. improve matching algorithm