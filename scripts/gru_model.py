from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Literal
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def build_vocab(tokenized_texts, max_vocab_size=10000, min_freq=1):
    """tokenized_texts (List[str]):
        A list of words (tokens) extracted from the text data. This could be a flattened list of all tokens from multiple documents or speeches.
        
        max_vocab_size (int, default=10,000):
        The maximum number of unique words to include in the vocabulary. Limiting the vocabulary size helps manage memory usage and computational efficiency, especially when dealing with large datasets.

        min_freq (int, default=1):
        The minimum frequency a word must have to be included in the vocabulary. This helps in excluding very rare words that might not contribute significantly to the model’s learning and can introduce noise."""

    counter = Counter(tokenized_texts)
    # Keep only the top common words
    most_common = counter.most_common(max_vocab_size)
    # <PAD> = Used to pad sequences to a uniform length, which is essential for batch processing in models.
    # <UNK> = Represents words that are not present in the vocabulary (i.e., out-of-vocabulary words). This helps the model handle unexpected or rare words gracefully during inference
    vocab = ['<PAD>', '<UNK>'] + [word for word, freq in most_common if freq >= min_freq]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    return word2idx


def encode_text(tokenized_text, word2idx):
    """
    Encodes a list of tokens into their corresponding indices.

    Args:
        tokenized_text (List[str]): A list of word tokens.
        word2idx (Dict[str, int]): A dictionary mapping words to indices.

    Returns:
        List[int]: A list of encoded word indices.
    """
    
    return [word2idx.get(word, word2idx['<UNK>']) for word in tokenized_text]


def encode_pos_tags(pos_tags_list, pos_tag2idx):
    """
    Encodes a list of POS tags into their corresponding indices without padding.

    Args:
        pos_tags_list (List[List[str]]): A list of POS tag sequences.
        pos_tag2idx (Dict[str, int]): A dictionary mapping POS tags to indices.

    Returns:
        List[List[int]]: A list of encoded POS tag sequences.
    """
    
    encoded = []
    for tags in pos_tags_list:
        encoded_tags = [pos_tag2idx.get(tag, pos_tag2idx['<PAD>']) for tag in tags]
        encoded.append(encoded_tags)
        
    return encoded


# Slice text into chunks
def slice_lists(row, window_size):
    n = len(row['encoded_text'])
    slices = []
    
    # Create slices of window_size
    for i in range(0, n, window_size):
        text_slice = row['encoded_text'][i:i + window_size]
        pos_slice = row['encoded_pos_tags'][i:i + window_size]
        label = row['label']
        slices.append([text_slice, pos_slice, label])
    
    return slices


# Define the Dataset
class SpeechDataset(Dataset):
    def __init__(self, texts, pos_tags, labels):
        self.texts = texts          # List of encoded texts
        self.pos_tags = pos_tags    # List of encoded POS tags
        self.labels = labels        # List of labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.pos_tags[idx], self.labels[idx]


# Define the collate function
def collate_fn(batch):
    """Batch Processing and Padding for Variable Length Sequences."""

    texts, pos_tags, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.int64)
    
    # Verify that each text and its POS tags have the same length
    # for i in range(len(texts)):
    #     assert len(texts[i]) == len(pos_tags[i]), f"Sample {i} has text length {len(texts[i])} and pos_tags length {len(pos_tags[i])}"
    
    # Pad sequences
    padded_texts = nn.utils.rnn.pad_sequence(
        [torch.tensor(text, dtype=torch.long) for text in texts], 
        batch_first=True, 
        padding_value=0
    )
    padded_pos_tags = nn.utils.rnn.pad_sequence(
        [torch.tensor(pt, dtype=torch.long) for pt in pos_tags], 
        batch_first=True, 
        padding_value=0
    )

    labels = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    
    return padded_texts, padded_pos_tags, lengths, labels

# Define GRU model
class SentimentGRU(nn.Module):
    def __init__(self,
                 hidden_dim, cell_state_info: Literal['ignore', 'add', 'linear'],
                 output_dim, n_layers, dropout_gru, dropout_h_c_states,
                 pad_idx, pos_pad_idx,
                 vocab_size, embedding_dim, 
                 pos_vocab_size, pos_embedding_dim,
                 pretrained_embeddings=None, freeze_embeddings=True, 
                 pos_embeddings=False):
        super(SentimentGRU, self).__init__()

        # Model parameters
        self.hidden_dim = hidden_dim
        self.cell_state_info = cell_state_info
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_gru = dropout_gru
        self.dropout_h_c_states = dropout_h_c_states

        # Padding
        self.pad_idx = pad_idx
        self.pos_pad_idx = pos_pad_idx

        # Embedding parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pos_vocab_size = pos_vocab_size
        self.pos_embedding_dim = pos_embedding_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.freeze_embeddings = freeze_embeddings
        self.pos_embeddings = pos_embeddings

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Combine word and POS embeddings
        self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_embedding_dim, padding_idx=self.pos_pad_idx)
        
        if self.pos_embeddings:
            combined_dim = self.embedding_dim + self.pos_embedding_dim
        else:
            combined_dim = self.embedding_dim

        # GRU layer
        self.gru = nn.GRU(
            combined_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout_gru if self.n_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout_h_c_states)

        # Fully connected layer to merge hidden states
        self.merge_hidden = nn.Linear(2*self.hidden_dim, self.hidden_dim)

        # Fully connected layer for output
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, text, pos_tags, text_lengths):
        # text: [batch size, sent_length]
        embedded = self.embedding(text)              # [batch size, sent_length, embedding_dim]

        # Concatenate word and POS embeddings
        if self.pos_embeddings:
            # pos_tags: [batch size, sent_length]
            pos_embedded = self.pos_embedding(pos_tags)
            # Check shapes before concatenation
            assert embedded.shape[0] == pos_embedded.shape[0], "Batch sizes do not match"
            assert embedded.shape[1] == pos_embedded.shape[1], "Sequence lengths do not match"

            combined = torch.cat((embedded, pos_embedded), dim=2) # [batch size, sent_length, combined_dim]
        else:
            combined = embedded

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            combined,
            text_lengths.cpu(),  # Ensure lengths are on CPU
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, hidden = self.gru(packed_embedded)

        # hidden: [num_layers, batch size, hidden_dim]
        # Take the last layer's hidden state
        # Apply dropout (maybe not needed)
        hidden = self.dropout_layer(hidden[-1, :, :])  # [batch size, hidden_dim]

        # Optionally handle hidden states for output
        if self.cell_state_info == 'linear':
            hidden = self.merge_hidden(torch.cat([hidden, hidden], dim=1))  # [batch size, 2*hidden_dim]
        elif self.cell_state_info == 'add':
            hidden = hidden + hidden  # Mimic cell state addition (redundant for GRU)
        else:
            hidden = hidden  # [batch size, hidden_dim]

        output = self.fc(hidden)  # [batch size, output_dim]

        return output
    
# Define Evaluation Metrics
def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculates precision, recall, F1-score, and ROC-AUC.
    
    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        y_probs (List[float]): Predicted probabilities.
        
    Returns:
        dict: Dictionary containing the metrics.
    """
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_probs)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    } 
# Define Function to Save Checkpoint
def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Saves the model and optimizer state.

    Args:
        state (dict): Contains model state_dict, optimizer state_dict, etc.
        filename (str): Path to save the checkpoint.
    """

    torch.save(state, filename)


# Define Function to Load Checkpoint
def load_checkpoint(filename, model, optimizer=None, device='cuda'):
    """
    Loads the model and optimizer state.

    Args:
        filename (str): Path to the checkpoint.
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load the state into.
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        
# Define Training Function for GRU Model
def train_model_gru(model, loader, criterion, optimizer, device='cuda'):
    """
    Trains the GRU model for one epoch.

    Args:
        model (nn.Module): The GRU model to train.
        loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.

    Returns:
        Tuple[float, dict]: Average loss and metrics for the epoch.
    """
    
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for texts, pos_tags, lengths, labels in loader:
        # Move tensors to device, except lengths
        texts = texts.to(device)
        pos_tags = pos_tags.to(device)
        labels = labels.to(device)
        
        # Label smoothing
        smoothing = torch.where(labels == 1, 0.1, 0.05)
        smoothed_labels = labels * (1 - smoothing) + smoothing / 2

        optimizer.zero_grad()

        outputs = model(texts, pos_tags, lengths)
        loss = criterion(outputs, smoothed_labels)  # labels
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Sigmoid to get probabilities
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = labels.detach().cpu().numpy()

        all_probs.extend(probs.flatten().tolist())
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(labels_np.flatten().tolist())

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    return epoch_loss / len(loader), metrics


# Define Evaluation Function for GRU Model
def evaluate_model_gru(model, loader, criterion, device='cuda'):
    """
    Evaluates the GRU model on a validation or test set.

    Args:
        model (nn.Module): The GRU model to evaluate.
        loader (DataLoader): DataLoader for the validation/test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        Tuple[float, dict]: Average loss and metrics for the evaluation.
    """
    
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for texts, pos_tags, lengths, labels in loader:
            # Move tensors to device, except lengths
            texts = texts.to(device)
            pos_tags = pos_tags.to(device)
            labels = labels.to(device)

            outputs = model(texts, pos_tags, lengths)  # [batch_size, 1]
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()

            # Sigmoid to get probabilities
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels_np = labels.detach().cpu().numpy()

            all_probs.extend(probs.flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels_np.flatten().tolist())

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    return epoch_loss / len(loader), metrics


# Define Test Function for GRU Model
def test_model_gru(model, loader, criterion, device='cuda'):
    """
    Tests the GRU model on the test set.

    Args:
        model (nn.Module): The trained GRU model.
        loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the test on.

    Returns:
        Tuple[float, dict]: Average loss and metrics for the test set.
    """
    
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for texts, pos_tags, lengths, labels in loader:
            # Move tensors to device, except lengths
            texts = texts.to(device)
            pos_tags = pos_tags.to(device)
            labels = labels.to(device)

            outputs = model(texts, pos_tags, lengths)  # [batch_size, 1]
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()

            # Sigmoid to get probabilities
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels_np = labels.detach().cpu().numpy()

            all_probs.extend(probs.flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels_np.flatten().tolist())

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    return epoch_loss / len(loader), metrics, (all_probs, all_preds, all_labels)

