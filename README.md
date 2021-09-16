# Recurrent_Neural_Network_applications

Applications of RNN in Sentiment Analysis (Many-to-One) and Music Generation trained on text audio notes (One-to-Many)

These sequence based tasks were implemented using Recurrent Neural Network with Long Short Term Memory (LSTM)

### A. English Sentiment Analysis (Many-to-One)

Sentiment analysis dataset is loaded from
https://ai.stanford.edu/ amaas/data/sentiment/ generated
by Andrew L. Maas, Raymond E. Daly, Peter T.
Pham, Dan Huang, Andrew Y. Ng, and Christopher
Potts. (2011). Learning Word Vectors for Sentiment
Analysis. The 49th Annual Meeting of the Association
for Computational Linguistics (ACL 2011).

The data is preprocessed before
vectorizing the test using a vocabulary trained on
tensorflow. This vectorized text is then used as
input to embedding layer which creates a trainable
lookup table to map index to vectors of the output
dimension and decrease dimensions of input
data. A bidirectional Lstm is used to retain context
from both directions and understand the sentiment
properly. Multiple layers are stacked to get better
result. Each layer pases its state to next except
the last LSTM layer. The output classification of
multi-sequence input is a single output determining
whether the sentiment was positive or negative.

### B. Audio-Input Music Generation (One-to-Many)

Music is generated using a data-set in the ABC
notation from Tunes compiled by Jack Campin
on http://www.campin.me.uk/ compiled music from
following artists: Embro: the hidden history of Edinburgh
in its music, Music of Dalkeith, Old Scottish
Flute Music, Aird’s Airs (A Selection of Scotch,
English, Irish and Foreign music).

The notation is cleaned and preprocessed before
creating a word vocabulary sorted and indexed to
use it in embedding layer. The training examples are
selected randomly from the set and the respective
target is right shifted as the model needs to classify
the next output, and create a sense of context while
moving forward. The model architecture during
training is a many-to-many style but generation is
one-to-many as a single music note is given which
in turn generates a output by understanding the
context in the music.



-----------------------------







Instructions to run 

-----Sentiment Analysis------

Senitment analysis program can be run using main file(Run main.py). Enter 0 to start sentiment analysis.

Then, enter 1 to retrain the model if needed(it is computationally expensive program due to LSTMs might take longer,
depending on availability of GPU libraries and GPU in the system). Enter number of epochs, trained model has 8 epochs.
After evaluating test loss and accuracy, a sample input to test model is given right after training.
Sample output for this is included in the submission. The model is saved in final_sentiment folder.
(sample output is marked with rectangles to help understanding it, as it is long)

Enter 0 to load pre-trained model without training. Enter 0 for best trained model or enter 1 to provide folder name (to help load the one you saved in train step).
5 text inputs are hardcoded to check prediction on the loaded model. To test more rerun.


Streamlit UI helps see the result as well (--optional). 
 INSTRUCTIONS TO RUN LOCALLY:
 1. RUN THE FOLLOWING COMMAND TO INSTALL STREAMLIT (if not already installed): pip install streamlit
 2. RUN THE FOLLOWING COMMAND  on the terminal TO RUN THIS FILE USING STREAMLIT:    streamlit run ui.py
 3. In browser click on load model, wait atleast 1 min to load and input boxes to highlight again.
 4. then enter text in input box and then its respective analyse sentiment button

Included sample input and output for streamlit.


-----Music notes Generation------

Music notes Generation program can be run using main file(Run main.py). Enter 1 to start music notes generation.

Then rest is same as previous.

At the end, it shows output and stores the output of indiviual songs which can be played in abc notation files. Trained model is stored in final_music folder.

We were unable to convert abc notation to midi or wav file and also there was no python library for it. We created our own but was not working properly.

Abc notation as text from final_music folder can be easily loaded to https://www.mandolintab.net/abcconverter.php converter online and check the generated music.
Included 3 sample midi file converted from abc which was trained and stored in final_music folder, which can be played on widows media player.
The abc notation file can be loaded to a conversion tool like https://www.nilsliberg.se/ksp/easyabc/ to convert to music and listen to the generated song.