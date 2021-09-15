
import io
import os
import regex as re
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensorflow GPU info verbose
import tensorflow as tf
import numpy as np
from playsound import playsound


#---Sentiment Analysis functions---##

# Downloading and preprocessing dataset for sentiment analysis from stanford edu datasets
# dataset have 2 labeled classes positive reviews and negative reviews of movies
def load_sentiment_dataset():
    # Large movie review Dataset from https://ai.stanford.edu/~amaas/data/sentiment/
    # Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    # Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
    dataset_1_location = os.path.join(os.getcwd(), 'aclImdb')
    dataset_1_train = os.path.join(dataset_1_location, 'train')
    ds_1_train = tf.keras.preprocessing.text_dataset_from_directory(dataset_1_train, subset='training', seed=1, batch_size=64, validation_split=0.2)
    ds_1_valid = tf.keras.preprocessing.text_dataset_from_directory(dataset_1_train, subset='validation', seed=1, batch_size=64, validation_split=0.2)
    ds_1_test = tf.keras.preprocessing.text_dataset_from_directory(os.path.join(dataset_1_location, 'test'), seed=1, batch_size=64)
    ds_1_train = ds_1_train.cache().shuffle(5000).prefetch(tf.data.AUTOTUNE) # shuffle in place
    ds_1_valid = ds_1_valid.cache().prefetch(tf.data.AUTOTUNE)
    ds_1_test = ds_1_test.cache().prefetch(tf.data.AUTOTUNE)
    for review, label in ds_1_train.take(1): # sample data
        print('Sample text sentiment: ', review.numpy()[:3])
        print('Sample label sentiment: ', label.numpy()[:3])
    return ds_1_train, ds_1_valid, ds_1_test


# text needs to be preprocessed to remove break tags and strip punctuation to process further
# this function needs to run when loading model
@tf.keras.utils.register_keras_serializable()
def sentiment_preprocess(data):
    data = tf.strings.lower(data)
    data = tf.strings.regex_replace(data, '<br />', ' ')
    # remove punctuation
    data = tf.strings.regex_replace(data,'[%s]' % re.escape(r'!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\''), '')
    return data


#---Music Generation functions---##

# this function helps select random input for the training of the given sequence length
def random_music_batch_inputs(music_length, music_vectors, music_sequence_length, seq_batch_size):
    # select start of the index
    start_index = np.random.choice(music_length - music_sequence_length, seq_batch_size)
    # input starts at random start index and the target to predict is shifted right to predict the next sequence
    seq_input = [music_vectors[current_index: current_index + music_sequence_length] for current_index in start_index]
    seq_target = [music_vectors[current_index+1: current_index + music_sequence_length+1] for current_index in start_index]
    return np.reshape(seq_input, [seq_batch_size, music_sequence_length]), np.reshape(seq_target, [seq_batch_size, music_sequence_length])


def music_model_arch(embed_size_music, embed_output_size_music, seq_batch):
    music_model = tf.keras.Sequential(
        [
            # embedding layer takes a positive indexed sequence and convert them into dense vectors to further feed the model
            # this layer creates a trainable lookup table to map index to vectors of the output dimension
            tf.keras.layers.Embedding(input_dim=embed_size_music, output_dim=embed_output_size_music, batch_input_shape =[seq_batch, None]),
            # lstm layer with 1024 units, with sigmoid recurrent activation and passing of each state active
            tf.keras.layers.LSTM(1024, activation='tanh', stateful=True, recurrent_activation='sigmoid', return_sequences=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(embed_size_music) # returns the probability of
        ]
    )
    return music_model


def loss_func_music(true_value, predicted):
    # a sparse categorical cross entropy loss is used since the prediction of next step in sequence is done by previous RNN
    sparse_cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_logits = sparse_cat_loss(true_value, predicted)
    return loss_logits


def apply_music_gradients(music_model, optimizer_music, true_value, predicted):
    with tf.GradientTape() as tape:
        # the input will be the randomly selected sequence and right shifted target for loss calculation.
        input_prediction = music_model(true_value)
        loss_logits = loss_func_music(predicted, input_prediction)
    # gradients are applied using adam optimizer
    gradients = tape.gradient(loss_logits, music_model.trainable_variables)
    optimizer_music.apply_gradients(zip(gradients, music_model.trainable_variables))
    return loss_logits


def vectorize_music_vocab(music_vocab_file):
    # get saved music vocab file, file is saved so it can used without loading entire program.
    text_to_list = music_vocab_file.strip().split('\n')
    # inserting space and \n that gets lost during vocab saving
    text_to_list.insert(0, ' ')
    text_to_list.insert(0, '\n')
    # indexing the characters in the vocabulary by their indices, use character to get index to use in embedding layer
    char_vector = {char:index for index, char in enumerate(text_to_list)}
    print()
    print("Vectorzed music Vocabulary: ")
    print(char_vector)
    # converting to array to use index to get character
    vector_char = np.array(text_to_list)
    return char_vector, vector_char, len(text_to_list)


def generate_music_notation(music_model_load, char_vector, vector_char, initial_key, generation_length = 2000):
    generated_music = []
    input_seq_gen = tf.expand_dims([char_vector[char] for char in initial_key], 0)
    for i in range(generation_length):
        # the predicted probability is fed as next sequence during generation while remebering the context of previous generation
        # the updated state of the RNN is passed to contain some sense of context.
        predicted_notation = music_model_load(input_seq_gen)
        predicted_indices = tf.random.categorical(tf.squeeze(predicted_notation, 0), num_samples=1)[-1,0].numpy()
        generated_music.append(vector_char[predicted_indices])
        input_seq_gen= tf.expand_dims([predicted_indices],0)
    return initial_key+''.join(generated_music)


def convert_to_song(gen_song, music_file_path):
    # the generated text is saved as abc file
    music_file = "{}.abc".format(music_file_path)
    with open(music_file, "w") as f:
        f.write(gen_song)
    # os.rename(music_file, pre + '.wav')
    # wav_file = music_file_path+'.wav'


def main():
    model_type = int(input("Press 0 for Sentiment Analysis, Press 1 for Music Generation: "))
    ###--------Sentiment Analysis--------###

    if model_type == 0:  # sentiment analysis
        train_sentiment_model = 0
        train_sentiment_model = int(input("Enter 1 to retrain Sentiment Model, Enter 0 to load pre-trained model and test on sample input: "))

        # ---Final Sentiment Analysis model was trained with 8 epochs and batch size of 64
        # using a Nvidia 1050ti GPU optimized tensorflow library on PyCharm
        if train_sentiment_model == 1:
            ds_1_train, ds_1_valid, ds_1_test = load_sentiment_dataset()
            # sentiment analysis model takes text in the form of vectors so text is vectorized using tensorflow textvectorization layer
            # this layer creates a vocabulary where vocabulary is trained on the review strings and the indices of words will be used to vectorize them.
            # the output will be padded to 400 to limit the review length to 400 (due to computation constraints)
            vocabulary_size_ds_1 = 10000
            train_reviews = ds_1_train.map(lambda review, label: review) # returns only reviews from train set
            word_vectors_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=sentiment_preprocess, max_tokens=vocabulary_size_ds_1, output_sequence_length=400)
            word_vectors_layer.adapt(train_reviews) # creates the vocabulary on the data to help with encoding text with its indices
            sentiment_sav_folder = input("Enter the name of folder to save sentiment analysis model data: ")
            epochs = int(input("Enter number of epochs (ideally 5-7)(time extensive, gpu will speed up): "))
            embed_size_senti, embed_output_size_senti = len(word_vectors_layer.get_vocabulary()), 64
            # creating model layers
            sentiment_model = tf.keras.Sequential(
                [
                    word_vectors_layer,
                    # embedding layer takes word indices created from textVectorization layer and uses a index-lookup mechanism
                    # to create sequences of vectors, basically embed the dimensions of vocabulary to a lower dimension, 64 here
                    tf.keras.layers.Embedding(input_dim=embed_size_senti, output_dim=embed_output_size_senti, mask_zero=True),
                    # a bidirectional lstm layer preserves context from both sides as sentence in english language can have different context depending upon grammar.
                    # bidirectional layers pass the output forward and backword from one timestep to input at next timestep.
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, recurrent_initializer='glorot_uniform')),
                    # a multi layer is used to improve the inference and training
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, recurrent_initializer='glorot_uniform')),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh', recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')),
                    tf.keras.layers.Dense(64, activation='relu'),
                    # droput layer to regularize and reduce overfitting.
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(1)
                ]
            )
            # binary cross entropy loss is used since the sentiment is binary.
            sentiment_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
            print("Sentiment Analysis Model Training....")
            sentiment_history = sentiment_model.fit(ds_1_train, epochs=epochs, validation_data=ds_1_valid)
            # positive if predictions greater than equal to zero otherwise negative
            # sample texts
            test_text = 'The movie was so bad. I would never go watch the movie with this actor going forward. Half of the time I was asleep'
            prediction = sentiment_model.predict(np.array([test_text]))
            print(test_text)
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)
            test_text = 'what a great movie, the scenes and direction was very cool'
            prediction = sentiment_model.predict(np.array([test_text]))
            print(test_text)
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)
            test_text = "I enjoy his company. And he is a good chef"
            prediction = sentiment_model.predict(np.array([test_text]))
            print(test_text)
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)
            test_text = "He is the worst. he discourages me and forces me to do things that I hate. He disappoints me."
            prediction = sentiment_model.predict(np.array([test_text]))
            print(test_text)
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)

            # evaluating the loss and accuracy of model on test set
            test_loss_1, test_acc_1 = sentiment_model.evaluate(ds_1_test)
            print('Test Loss:', test_loss_1)
            print('Test Accuracy:', test_acc_1)

            test_input_text = input("Enter a Sample text to predict sentiment: ")
            prediction = sentiment_model.predict(np.array([test_input_text]))
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)
            print("saving---")
            # save the model and it architecture to load during streamlit or testing.
            sentiment_model.save(os.path.join(os.getcwd(), sentiment_sav_folder))
            print("Model saved under given folder, restart program to run model by entering 0")

        # to load the model
        elif train_sentiment_model == 0:
            train_type = int(input("Enter 0 for best trained model, else enter 1 to find from a saved folder trained previously: "))
            if train_type == 1:
                sentiment_folder_name = input("Enter folder name: ")
            else:
                sentiment_folder_name = "final_sentiment"
            print("Please wait for model to load....")
            sentiment_model_load = tf.keras.models.load_model(os.path.join(os.getcwd(), sentiment_folder_name))
            test_input_text = input("Enter a Sample text to predict sentiment: ")
            prediction = sentiment_model_load.predict(np.array([test_input_text]))
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)

            test_input_text = input("Enter a Sample text to predict sentiment: ")
            prediction = sentiment_model_load.predict(np.array([test_input_text]))
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)

            test_input_text = input("Enter a Sample text to predict sentiment: ")
            prediction = sentiment_model_load.predict(np.array([test_input_text]))
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)

            test_input_text = input("Enter a Sample text to predict sentiment: ")
            prediction = sentiment_model_load.predict(np.array([test_input_text]))
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)

            test_input_text = input("Enter a Sample text to predict sentiment: ")
            prediction = sentiment_model_load.predict(np.array([test_input_text]))
            print("Sample Prediction(if prediction >=0 sentiment +ve else -ve): ", prediction)
            print("Restart to start again!")

    ###--------Music Notes Generation--------###

    if model_type == 1:  # music notes generation
        tf.keras.backend.clear_session()
        train_music_model = 0
        train_music_model = int(input("Enter 1 to retrain Music Model, Enter 0 to load pre-trained music model and test on sample input: "))

        # ---Final music generation model was trained with 5000 epochs and batch size of 64
        # using a Nvidia 1050ti GPU optimized tensorflow library on PyCharm
        if train_music_model == 1:
            # dataset from Tunes compiled by Jack Campin on http://www.campin.me.uk/ compiled music from following artists:
            # Embro: the hidden history of Edinburgh in its music, Music of Dalkeith, Old Scottish Flute Music, Aird's Airs
            # (A Selection of Scotch, English, Irish and Foreign music)
            music_dataset = io.open(os.path.join(os.getcwd(), 'abc_mus.txt'), encoding='UTF-8').read().strip().split('\n')
            music_lines = ''
            for i in range(len(music_dataset)):
                if music_dataset[i].find("Z:") != 0 and music_dataset[i].find("B:") != 0 and music_dataset[i].find("N:") != 0:
                    music_lines = music_lines + "\n" + music_dataset[i]

            preprocess_music = re.findall('(^|\n\n)(.*?)\n\n', music_lines, flags=re.DOTALL)
            midi = [song[1] for song in preprocess_music]
            print("Found {} songs in text".format(len(midi)))
            print()
            print("Sample song in abc notation file: ")
            print(midi[1])
            all_midi = "\n\n".join(midi)
            # cannot override old folder
            music_sav_folder = str(input("Enter the name of folder to save music generation model data: "))
            os.mkdir(os.path.join(os.getcwd(), music_sav_folder))
            # generate a vocabulary of words and use its indexes as lookup during embedding to give numerical representation to feed to next layer.
            music_vocab = sorted(set(all_midi))
            with open(os.path.join(os.getcwd(), music_sav_folder, 'music_vocab.txt'), "w") as f:
                f.write('\n'.join(music_vocab))
            # load index to word mapping
            char_vector, vector_char, vocab_length = vectorize_music_vocab(io.open(os.path.join(os.getcwd(), music_sav_folder, 'music_vocab.txt')).read())
            # use vectorization to find index of our dataset
            music_vectors = np.array([char_vector[char] for char in all_midi])
            music_length = len(music_vectors)-1

            epochs = int(input("Enter number of epochs (ideally 2000-4000)(time extensive, gpu will speed up): "))
            # size of input and output embedding layer
            embed_size_music, embed_output_size_music, seq_len, seq_batch = len(music_vocab), 256, 100, 64
            music_model = music_model_arch(embed_size_music, embed_output_size_music, seq_batch)
            optimizer_music = tf.keras.optimizers.Adam()
            clock = time.time()
            # training step
            print("Music Generation Model Training....")
            for epoch in range(epochs):
                input_seq, target_seq = random_music_batch_inputs(music_length, music_vectors, seq_len, seq_batch)
                current_batch_loss = apply_music_gradients(music_model, optimizer_music, input_seq, target_seq)
                if epoch % 400 == 0:
                    print(f'Epoch no. {epoch+1} Loss = {current_batch_loss.numpy():.3f}')

            print(f'Total time for all Epochs: {time.time()-clock:.2f} sec\n')
            music_model.save_weights(os.path.join(os.getcwd(), music_sav_folder, 'weights'))
            print()
            # inference step
            # reload the model but with new input size as only one input is given as seed for text generation
            music_model_load = music_model_arch(vocab_length, 256, 1)
            music_model_load.load_weights(os.path.join(os.getcwd(), music_sav_folder, 'weights'))
            music_model_load.build(tf.TensorShape([1, None]))
            music_key = str(input("Enter key to generate song with: "))
            gen_text = generate_music_notation(music_model_load, char_vector, vector_char, music_key, generation_length = 2000)
            print("Generated music in abc notation: ")
            print(gen_text)
            # preprocessing generated text to store in abc format which can be converted to midi to play songs.
            snippet = re.findall('(^|\n\n)(.*?)\n\n', gen_text, flags=re.DOTALL)
            gen_songs = [song[1] for song in snippet]

            # unable to run on windows due to unwanted behaviour of abc to midi file conversion can only run on linux, unfortunately
            # therefore only saving abc file which can be converted using online tools or a python exe file in submission
            # file or text can be easily copied on https://www.mandolintab.net/abcconverter.php to generate music
            for song_number, song in enumerate(gen_songs):
                music_file_path = os.path.join(os.getcwd(), music_sav_folder, str(song_number)+music_key)
                convert_to_song(song, music_file_path)

        # loading pre-trained models
        elif train_music_model == 0:
            train_type = input("Enter 0 for best trained model, else enter 1 to find from a saved folder trained previously: ")
            if train_type == 1:
                music_folder_name = str(input("Enter folder name: "))
            else:
                music_folder_name = "final_music"
            load_vocab_test = io.open(os.path.join(os.getcwd(), music_folder_name, 'music_vocab.txt')).read()
            char_vector, vector_char, vocab_length = vectorize_music_vocab(load_vocab_test)
            music_model_load = music_model_arch(vocab_length, 256, 1)
            music_model_load.load_weights(os.path.join(os.getcwd(), music_folder_name, 'weights'))
            music_model_load.build(tf.TensorShape([1, None]))
            music_key = str(input("Enter key to generate song with: "))
            gen_text = generate_music_notation(music_model_load, char_vector, vector_char, music_key, generation_length = 2000)
            print("Generated music in abc notation: ")
            print(gen_text)
            snippet = re.findall('(^|\n\n)(.*?)\n\n', gen_text, flags=re.DOTALL)
            gen_songs = [song[1] for song in snippet]


            # unable to run on windows due to unwanted behaviour of abc to midi file conversion can only run on linux, unfortunately
            # therefore only saving abc file which can be converted using online tools or a python exe file in submission
            for song_number, song in enumerate(gen_songs):
                music_file_path = os.path.join(os.getcwd(), music_folder_name, str(song_number)+music_key)
                convert_to_song(song, music_file_path)


if __name__ == '__main__':
    main()
