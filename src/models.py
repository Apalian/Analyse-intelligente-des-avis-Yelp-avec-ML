import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron pour la classification de texte.

    Architecture :
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> FC(n_classes)

    Ameliorations :
        - Initialisation Kaiming pour les couches lineaires
        - BatchNorm pour stabiliser l'entrainement
    """
    def __init__(self, input_dim, n_classes, hidden_dims=(256, 128), dropout=0.3):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


class TextCNN(nn.Module):
    """
    CNN pour la classification de texte (Yoon Kim, 2014).

    Architecture :
        Embedding -> Conv1D (multi-filtres) + BatchNorm -> MaxPool -> FC -> Output

    """
    def __init__(self, vocab_size, embed_dim, n_classes,
                 n_filters=128, filter_sizes=(3, 4, 5), dropout=0.4):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs),
                nn.BatchNorm1d(n_filters),
                nn.ReLU()
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            c = conv(embedded)
            c = c.max(dim=2).values
            conv_outputs.append(c)

        out = torch.cat(conv_outputs, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
