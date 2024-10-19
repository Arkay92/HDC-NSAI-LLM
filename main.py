import numpy as np
import networkx as nx
from sympy import cos
import torch
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, recall_score
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import download, pos_tag
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
import random
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision training
set_global_policy('mixed_float16')

# Load the HelpSteer dataset
dataset = load_dataset("nvidia/HelpSteer")
train_data = dataset['train']
val_data = dataset['validation']

# Download NLTK data
download('punkt')
download('stopwords')

# Load the pre-trained GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

# Set the padding token to the EOS token
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Hyperparameters
DIM = 2500  # Hypervector dimension size
EXPERIENCE_BUFFER_SIZE = 200  # Buffer size for memory
DECAY_RATE = 0.98  # Memory decay rate
TIMESTEPS = 1  # Each sentence is treated as one timestep
INITIAL_LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
L2_REG = 1e-4  # L2 regularization
PATIENCE = 15
KFOLDS = 10  # Number of folds for cross-validation
BATCH_SIZE = 256
EPOCHS = 100  # Episodes
AUGMENTATION_PROB = 0.3  # Probability of applying augmentation

# Early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch > 10:  # Start decaying after 10 epochs
        return lr * 0.8
    return lr

# Convert a numpy array to hexadecimal representation
def to_hex(arr):
    return [hex(int(x)) for x in arr]

# Model definition with optional architecture (LSTM, GRU, Bidirectional LSTM)
def neural_network_module(input_dim, timesteps=TIMESTEPS, architecture="LSTM", n_units=64, dropout_rate=DROPOUT_RATE):
    nn_model = Sequential()
    if architecture == "BERT":
        nn_model.add(TFBertModel.from_pretrained('bert-base-uncased'))
    elif architecture == "LSTM":
        nn_model.add(LSTM(n_units, activation='relu', return_sequences=False))
    elif architecture == "GRU":
        nn_model.add(GRU(n_units, activation='relu', return_sequences=False))
    elif architecture == "BidirectionalLSTM":
        nn_model.add(Bidirectional(LSTM(n_units, activation='relu', return_sequences=False)))

    nn_model.add(Dropout(dropout_rate))
    nn_model.add(Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)))
    nn_model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return nn_model

# Generate GPT-2 embeddings
def generate_gpt2_embeddings(sentences):
    inputs = gpt2_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        embeddings = outputs.last_hidden_state
    return embeddings.numpy()

# Dynamic Paraphrasing with POS tags using WordNet
def paraphrase_sentence(sentence):
    words = sentence.split()
    new_sentence = words[:]

    if random.random() < AUGMENTATION_PROB:
        tagged_words = pos_tag(words)
        nouns_verbs = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('V')]
        if nouns_verbs:
            word_to_replace = random.choice(nouns_verbs)

            # Find synonyms using WordNet
            synonyms = []
            for syn in wordnet.synsets(word_to_replace):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())

            # Replace the word with a synonym if available
            if synonyms:
                paraphrase = random.choice(synonyms)
                new_sentence = [paraphrase if word == word_to_replace else word for word in words]

    return ' '.join(new_sentence)

# Enhanced hypervector representation
def holographic_reduced_representation(tokens):
    # Preallocate arrays
    token_hvs = np.random.choice([-1, 1], size=(len(tokens), DIM))
    pos_hvs = np.roll(np.eye(DIM), np.arange(len(tokens)), axis=0)  # Create position hypervectors in one step

    # Use matrix multiplication instead of a loop to compute the hypervector
    sentence_hv = np.sum(token_hvs * pos_hvs[:len(tokens)], axis=0)  # Element-wise multiplication and sum

    return to_hex(sentence_hv)  # Convert to hex before returning

# Knowledge graph
def create_expanded_knowledge_graph():
    G = nx.Graph()
    G.add_edges_from([
        ("Entity", "Attribute"),
        ("Action", "Consequence"),
        ("Agent", "Action"),
        ("Input", "Output"),
        ("Goal", "Obstacle"),
        ("State", "Transition"),
        ("Cause", "Effect"),
        ("Premise", "Conclusion"),
        ("Condition", "Result"),
        ("Request", "Response"),
        ("Topic", "Subtopic"),
        ("Concept", "Related Concept"),
    ])
    return G

# Symbolic reasoning step based on hypervector and knowledge graph
def symbolic_reasoning_step(hypervector, knowledge_graph):
    if np.sum([int(val, 16) for val in hypervector[:10]]) > 0 and knowledge_graph.has_edge("Entity", "Attribute"):
        return float(cos(np.pi / 6))
    return 1.0

# Token prediction with HDC and symbolic reasoning
def predict_next_token(knowledge_graph, input_text, architecture="LSTM"):
    words = word_tokenize(input_text)
    hypervector = holographic_reduced_representation(words)
    symbolic_value = symbolic_reasoning_step(hypervector, knowledge_graph)

    model = neural_network_module(input_dim=DIM, timesteps=TIMESTEPS, architecture=architecture)
    transformed_hv = (np.array([int(val, 16) for val in hypervector]) * symbolic_value).reshape((1, TIMESTEPS, DIM))

    prediction = model.predict(transformed_hv)
    next_token = "explore" if prediction > 0.5 else "refine"
    return next_token

# Cross-validation with Stratified K-Folds
def cross_validate_model(X, y, k=KFOLDS, architecture="LSTM"):
    models = []  # To keep track of trained models
    class_counts = np.bincount(y)
    min_class_count = np.min(class_counts)
    k = min(k, min_class_count)  # Ensure k doesn't exceed the smallest class size

    if k < 2:
        print("Not enough samples in each class for cross-validation.")
        return

    stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies, fold_f1s, fold_recalls = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Reshape X to 2D for resampling
        X_train_2D = X_train.reshape((X_train.shape[0], -1))
        X_test_2D = X_test.reshape((X_test.shape[0], -1))

        # Apply SMOTE or RandomOverSampler based on class counts
        minority_class_count = np.min(np.bincount(y_train))
        if minority_class_count >= 2:
            k_neighbors = min(5, minority_class_count - 1)
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train_2D, y_train)
        else:
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train_2D, y_train)
            print(f"Using RandomOverSampler for fold {fold_idx + 1} due to insufficient samples for SMOTE.")

        # Reshape back to 3D for LSTM input
        X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], TIMESTEPS, DIM))
        X_test_3D = X_test_2D.reshape((X_test.shape[0], TIMESTEPS, DIM))

        # Train the model
        model = neural_network_module(input_dim=DIM, timesteps=TIMESTEPS, architecture=architecture)
        model.fit(X_train_resampled, y_train_resampled, epochs=EPOCHS,
                  callbacks=[LearningRateScheduler(lr_scheduler), early_stopping, model_checkpoint, reduce_lr],
                  batch_size=BATCH_SIZE, verbose=0)

        # Save the trained model for this fold
        models.append(model)

        # Evaluate on the test set
        y_pred = model.predict(X_test_3D)
        y_pred = (y_pred.flatten() > 0.5).astype(int)

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)

        fold_accuracies.append(accuracy)
        fold_f1s.append(f1)
        fold_recalls.append(recall)

        print(f"Fold {fold_idx + 1}: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}, Recall={recall:.4f}")

    print(f"\nCross-Validation Results:")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Average F1 Score: {np.mean(fold_f1s):.4f}")
    print(f"Average Recall: {np.mean(fold_recalls):.4f}")

    return models  # Return the list of trained models

# Data Augmentation by word replacement
def augment_sentence(sentence, augmentation_prob=AUGMENTATION_PROB):
    words = sentence.split()
    if random.random() < augmentation_prob:
        swap_idx = random.randint(0, len(words) - 1)
        words[swap_idx] = random.choice(["AI", "ML", "neural", "networks", "deep", "learning"])
    return ' '.join(words)

# Dataset preparation
def preprocess_data(data, augmentation=False):
    X = []
    y = []
    for example in data:
        prompt = example['prompt']
        response = example['response']
        combined_text = prompt + " " + response
        tokens = word_tokenize(combined_text)
        if augmentation:
            tokens = word_tokenize(augment_sentence(combined_text))  # Apply augmentation
        hypervector = holographic_reduced_representation(tokens)
        X.append(hypervector)
        # Binary classification: 1 if helpfulness score > 2, else 0
        y.append(1 if example['helpfulness'] > 2 else 0)
    return np.array(X).reshape((len(X), TIMESTEPS, DIM)), np.array(y)

if __name__ == '__main__':
    knowledge_graph = create_expanded_knowledge_graph()

    # Prepare the HelpSteer dataset
    X_train, y_train = preprocess_data(train_data, augmentation=True)  # Training with augmentation
    X_val, y_val = preprocess_data(val_data)  # Validation data

    # Distill knowledge using GPT-2 with batch processing
    gpt2_sentences = [example['prompt'] + " " + example['response'] for example in train_data]
    gpt2_embeddings = generate_gpt2_embeddings(gpt2_sentences, batch_size=16)  # Adjust batch size as needed
    gpt2_embeddings = gpt2_embeddings[:, :DIM]  # Ensure embeddings match the DIM
    X_train = np.concatenate((X_train, gpt2_embeddings), axis=0)  # Combine with embeddings

    # Perform cross-validation and test with Bidirectional LSTM
    trained_models = cross_validate_model(X_train, y_train, architecture="BidirectionalLSTM")

    # Test prediction with the last trained model (or modify as needed)
    refined_token = predict_next_token(knowledge_graph, "I need assistance with AI.", architecture="BidirectionalLSTM")
    print(f"Refined predicted token: {refined_token}")

    # Load the best model after training for inference
    best_model = load_model('best_model.keras')
    print("Best model loaded for inference.")
