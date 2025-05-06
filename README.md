# Ai202L-Project

### 🏛️ Roman Numeral Classifier (I–X)
This Flask web app lets you upload or draw a Roman numeral (I to X), and it'll predict the value using a trained neural network model. It uses a main CNN classifier and a binary subnet model to fix common confusion between 'II' and 'V'.

### ✨ How It Works
#### 🧠 Main Model (Except2.keras)
A CNN trained to classify handwritten Roman numerals I to X.

Output: 10-class softmax (I, II, III, IV, V, VI, VII, VIII, IX, X).

Input: 28×28 grayscale image (preprocessed and centered).

Architecture: basic Conv2D + MaxPooling + Dense layers.

⚠️ Known Issue
'II' is always predicted as 'V'.

#### 🧩 Subnet Model (2-5-9th.keras)
A binary classifier to distinguish between 'II' and 'V' more precisely.

Activated only when main model predicts 'V'.

Output: sigmoid → 1 = 'II', 0 = 'V'.

Architecture: small CNN (Conv2D + Dense + Dropout).

#### 🔁 Preprocessing
Base64 input → grayscale PIL image.

Crops whitespaces for main model.

Resizes to 28×28.

Normalized to [0, 1].

Optional augmentation (e.g., flip or rotate for subnet).

#### 🧪 Model Training (TL;DR)
Trained on a synthetic + real dataset of handwritten numerals.

Main model: categorical crossentropy.

Subnet: binary crossentropy.

Augmentation: flipping, rotation (optional).

