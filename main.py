import os
from preprocessing import prep
import lstm
import numpy as np
from sklearn.utils import shuffle


def cleanData(doc, kolom):
    data_clean = prep.removeUnicode(doc, kolom)
    data_clean = prep.removeMention(data_clean, kolom)
    data_clean = prep.removeLink(data_clean, kolom)
    data_clean = prep.removeNumberAndSymbol(data_clean, kolom)
    data_clean = prep.removeamp(data_clean, kolom)
    return data_clean


if __name__ == "__main__":

    training = True
    run_name = "text_classifier_lstm"

    train_path = "./data/train"
    validation_path = "./data/dev"
    test_path = "./data/test"
    kolom = "Tweet"
    kolom_label = "Affect Dimension"

    max_len_pad = 10
    embed_size = 128
    n_hidden_units = 1024
    n_lstm_layers = 8
    dropout_ratio = 0.2

    n_epoch = 200
    batch_size = 32

    n_kelas = 4

    # Data Training
    data_train = prep.importData(train_path, "\t")
    data_train_clean = cleanData(data_train, kolom)
    data_train_clean = shuffle(data_train_clean)
    # Data Validation
    data_validation = prep.importData(validation_path, "\t")
    data_validation_clean = cleanData(data_validation, kolom)
    data_validation_clean = shuffle(data_validation_clean)
    # Data Testing
    data_test = prep.importData(test_path, "\t")
    data_test_clean = cleanData(data_test, kolom)
    data_test_clean = shuffle(data_test_clean)

    vocab = prep.createVocab(data_train_clean[kolom])
    vocab_size = len(vocab)

    # One Hot encode label train
    LABEL_ENCODER_TRAIN, label_train_one_hot = prep.oneHotEncode(
        data_train_clean[kolom_label])
    # One Hot encode label validation
    LABEL_ENCODER_VALIDATION, label_validation_one_hot = prep.oneHotEncode(
        data_validation_clean[kolom_label])
    # One Hot encode label test
    LABEL_ENCODER_TEST, label_test_one_hot = prep.oneHotEncode(
        data_test_clean[kolom_label])

    # padding word train
    word_padded_train = lstm.wordEmbedding(
        vocab_size, max_len_pad, data_train_clean, kolom)
    # padding word validation
    word_padded_validation = lstm.wordEmbedding(
        vocab_size, max_len_pad,  data_validation_clean, kolom)
    # padding word test
    word_padded_test = lstm.wordEmbedding(
        vocab_size, max_len_pad, data_test_clean, kolom)

    if(training):
        model = lstm.defineModel(
            vocab_size,
            embed_size,
            dropout_ratio,
            n_hidden_units,
            word_padded_train.shape,
            n_kelas,
            n_lstm_layers
        )

        model = lstm.runModel(
            model,
            word_padded_train,
            label_train_one_hot,
            n_epoch,
            batch_size,
            run_name,
            word_padded_validation,
            label_validation_one_hot
        )
    else:
        model = lstm.importModel("./models/namamodel.h5")

    prediction = lstm.predictVal(model, word_padded_test, label_test_one_hot)

    val_str = prep.OneHoteDecode(LABEL_ENCODER_TEST, label_test_one_hot)
    pred_str = prep.OneHoteDecode(LABEL_ENCODER_TEST, prediction)
    with open("hasil.txt", "w") as f:
        for real, pred, text in zip(val_str, pred_str, data_validation_clean):
            hasil = str(real) + "\t=>\t"+str(pred) +": " +str(text)
            f.write(hasil)
