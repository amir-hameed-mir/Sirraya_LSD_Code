# Layer-wise Semantic Dynamics: Complete Documentation
## From Beginner to Advanced - Understanding Hallucination Detection in LLMs

---

# PART 1: FOUNDATIONS
## Understanding the Core Concepts

### 1.1 What is Layer-wise Semantic Dynamics?

Imagine you're reading a book, and with each page, your understanding of the story evolves. Similarly, when a Large Language Model (LLM) processes text, its "understanding" evolves as the information flows through its neural network layers. **Layer-wise Semantic Dynamics** is the study of how this understanding changes from the input layer to the output layer.

Think of it this way:
- **Layer 1**: The model sees raw words (like seeing individual letters on a page)
- **Middle Layers**: The model starts forming concepts (like understanding words form sentences)
- **Final Layers**: The model has full semantic understanding (like comprehending the complete story)

### 1.2 Why Does This Matter for Hallucination Detection?

**Hallucinations** in LLMs occur when the model generates factually incorrect information while sounding confident. For example:
- **Correct**: "The Eiffel Tower is in Paris, France"
- **Hallucination**: "The Eiffel Tower is in Rome, Italy"

The key insight of Layer-wise Semantic Dynamics is that **truthful and hallucinated information follow different "paths"** through the neural network layers. By analyzing these paths (trajectories), we can detect when the model is hallucinating.

### 1.3 The Core Idea in Simple Terms

```
INPUT: "The Eiffel Tower is in ___"
       ↓
LAYER 1: [Raw word embeddings]
       ↓
LAYER 2: [Basic syntax understood]
       ↓
LAYER 3: [Concepts forming]
       ↓
LAYER 4: [Semantic understanding]
       ↓
LAYER 5: [Final prediction]
       ↓
OUTPUT: "Paris" (factual) OR "Rome" (hallucination)
```

**What we discovered**: Factual information tends to converge to a stable representation early, while hallucinations show unstable, oscillating patterns through the layers.

---

# PART 2: THE ARCHITECTURE - Building Block by Block

## 2.1 System Overview

Our system consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────┐
│                 DATA LAYER                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Synthetic  │  │ TruthfulQA │  │ Custom     │        │
│  │ Data       │  │ Dataset    │  │ Datasets   │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                 MODEL LAYER                              │
│  ┌────────────────────┐  ┌────────────────────┐        │
│  │ LLM (GPT-2, etc.)  │  │ Truth Encoder      │        │
│  │ Extracts hidden    │  │ (Sentence-BERT)    │        │
│  │ states from all    │  │ Encodes truth      │        │
│  │ layers             │  │ statements         │        │
│  └────────────────────┘  └────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              PROJECTION LAYER                            │
│  ┌────────────────────────────────────┐                 │
│  │ Neural Networks that project both   │                 │
│  │ embeddings into a SHARED SPACE      │                 │
│  │ where we can compare them           │                 │
│  └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              ANALYSIS LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │Feature   │  │Statistical│  │Visual-   │             │
│  │Extraction│  │Analysis   │  │ization   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

## 2.2 Understanding Embeddings and Hidden States

### What is an Embedding?
An **embedding** is a numerical representation of text - essentially converting words into numbers that computers can process.

Think of it like this:
- **Word**: "cat"
- **Embedding**: [0.2, -0.5, 0.8, 0.1, ...] (a vector of numbers)

Each number captures some aspect of meaning:
- First dimension might represent "animal-ness"
- Second might represent "size"
- Third might represent "domestic vs wild"

### Hidden States Through Layers

When text passes through each layer of a neural network, its embedding transforms:

```
Layer 1: [0.2, -0.5, 0.8, 0.1] → Basic word meaning
Layer 2: [0.3, -0.4, 0.7, 0.2] → Syntax understood
Layer 3: [0.5, -0.3, 0.6, 0.3] → Context incorporated
Layer 4: [0.7, -0.2, 0.5, 0.4] → Semantic understanding
```

## 2.3 The Shared Space Concept

The key innovation: we project both the LLM's hidden states AND truth embeddings into a **shared space** where we can directly compare them.

```
LLM Space                    Shared Space                Truth Space
    │                             │                           │
    │     ┌─────────────────┐     │     ┌─────────────────┐   │
    │────▶│ Hidden Projector│────▶│◀────│ Truth Projector │◀──│
    │     └─────────────────┘     │     └─────────────────┘   │
    │                             │                           │
    │         Now we can measure  │                           │
    │         cosine similarity   │                           │
    │         between them!       │                           │
```

**Why is this powerful?** Because we train these projectors to map semantically similar content to nearby points in the shared space.

---

# PART 3: DEEP DIVE INTO KEY COMPONENTS

## 3.1 HiddenStatesExtractor - The "Eyes" of the System

### What It Does
Extracts the internal representations (hidden states) from every layer of a language model.

### How It Works
```python
class HiddenStatesExtractor:
    def get_hidden_states(self, texts):
        # For each text, we get:
        # - Input text: "The Earth orbits the Sun"
        # - Output: Tensor of shape [layers, hidden_dimension]
        #   where layers = number of model layers (e.g., 12 for GPT-2)
        #   and hidden_dimension = size of each layer (e.g., 768)
```

### Visual Understanding

```
Text: "The Earth orbits the Sun"
       ↓
Tokenizer → [101, 2054, 2567, 5892, 1012]  (token IDs)
       ↓
Embedding Layer → [batch, seq_len, 768]     (word embeddings)
       ↓
Layer 1 → [batch, seq_len, 768]             (first transformation)
       ↓
Layer 2 → [batch, seq_len, 768]             (deeper understanding)
       ↓
... (through all layers)
       ↓
Layer 12 → [batch, seq_len, 768]            (final representation)
       ↓
Mean Pooling → [batch, 768] per layer       (average across sequence)
       ↓
Stack → [batch, layers, 768]                 FINAL OUTPUT
```

### Why Mean Pooling?
Instead of using all token representations (which vary in length), we average them:
- Each token: "The" [0.1, 0.2, ...], "Earth" [0.3, 0.1, ...], "orbits" [0.2, 0.4, ...]
- Mean: [(0.1+0.3+0.2)/3, (0.2+0.1+0.4)/3, ...]
- Result: One vector representing the entire sentence

## 3.2 TruthEncoder - The "Oracle"

### What It Does
Encodes ground truth statements into embeddings that represent what factual information should look like.

### Why Sentence Transformers?
Sentence Transformers are specifically trained to create embeddings where semantically similar sentences are close together in vector space.

### Example:
```python
truth_encoder.encode_batch(["Earth revolves around the Sun"])
# Returns: [0.23, -0.56, 0.89, ...] (normalized embedding)

truth_encoder.encode_batch(["Earth orbits the Sun"])  
# Returns: [0.22, -0.55, 0.88, ...] (very similar!)
```

## 3.3 Projection Heads - The "Translators"

### Architecture Deep Dive

```python
def build_enhanced_projection_heads(hidden_dim, truth_dim, shared_dim):
    # Hidden states projector (LLM → Shared Space)
    hidden_proj = nn.Sequential(
        # Layer 1: Expand dimensions
        nn.Linear(hidden_dim, shared_dim * 4),  # e.g., 768 → 1024
        nn.GELU(),  # Activation function (like ReLU but smoother)
        nn.Dropout(0.2),  # Prevent overfitting
        nn.LayerNorm(shared_dim * 4),  # Stabilize training
        
        # Layer 2: Compress
        nn.Linear(shared_dim * 4, shared_dim * 2),  # 1024 → 512
        nn.GELU(),
        nn.Dropout(0.1),
        nn.LayerNorm(shared_dim * 2),
        
        # Layer 3: Final projection
        nn.Linear(shared_dim * 2, shared_dim),  # 512 → 256
        nn.LayerNorm(shared_dim)
    )
```

### Why This Architecture?
1. **Expansion then compression** (like an autoencoder) helps learn better representations
2. **GELU activation** handles negative values better than ReLU
3. **LayerNorm** stabilizes training by normalizing activations
4. **Dropout** prevents overfitting by randomly turning off neurons

## 3.4 Contrastive Learning - The "Teaching Method"

### The Core Training Concept

We train the system using **contrastive learning** - teaching it to pull similar pairs together and push dissimilar pairs apart.

```
Training Data Format:
[
    ("text", "truth", "label"),
    ("Earth orbits Sun", "Earth revolves around Sun", "factual"),      # Positive pair
    ("Earth orbits Moon", "Earth revolves around Sun", "hallucination") # Negative pair
]
```

### The Contrastive Loss Function

```python
def enhanced_contrastive_loss(cos_sim, labels, margin=0.5):
    # Positive pairs (factual): want cos_sim close to 1
    pos_loss = (1 - cos_sim[pos_mask]).pow(2).mean()
    
    # Negative pairs (hallucination): want cos_sim <= -margin
    neg_loss = F.relu(cos_sim[neg_mask] + margin).pow(2).mean()
    
    return 0.5 * pos_loss + 0.5 * neg_loss
```

### Visual Understanding of the Loss

```
BEFORE TRAINING:
Positive pair similarity: 0.3 (too low)
Negative pair similarity: 0.2 (too high - they're too close!)
Loss: High

AFTER TRAINING:
Positive pair similarity: 0.9 ✓ (pulled together)
Negative pair similarity: -0.6 ✓ (pushed apart)
Loss: Low

Goal: Maximize distance between classes in shared space
```

---

# PART 4: FEATURE ENGINEERING - What We Measure

## 4.1 Trajectory Features Explained

### The Alignment Trajectory

For each sample, we get a sequence of alignment scores across layers:

```
Layer 1: 0.1  (barely aligned with truth)
Layer 2: 0.3  (slightly better)
Layer 3: 0.6  (getting there)
Layer 4: 0.8  (good alignment)
Layer 5: 0.85 (peak alignment)
Layer 6: 0.82 (slight drop)
... and so on
```

### Key Features We Extract

#### 1. **Final Alignment** (last layer score)
- **What it measures**: How aligned is the final output with truth?
- **Intuition**: Factual statements should end with high alignment
- **Range**: -1 to 1 (higher is better for factual)

#### 2. **Mean Alignment** (average across layers)
- **What it measures**: Overall consistency of alignment
- **Intuition**: Factual statements maintain good alignment throughout
- **Hallucinations**: May show low alignment from early layers

#### 3. **Max Alignment** (peak score)
- **What it measures**: Best alignment achieved at any layer
- **Intuition**: Even if final output is good, when did it peak?
- **Interesting pattern**: Hallucinations sometimes peak early then collapse

#### 4. **Convergence Layer** (where max occurs)
- **What it measures**: Which layer achieves highest alignment
- **Intuition**: Factual info converges earlier and stabilizes
- **Pattern**: Factual: Layer 3-4, Hallucination: Layer 5-6 (late peaks)

#### 5. **Stability** (variance in last 3 layers)
- **What it measures**: How stable is the representation at the end
- **Intuition**: Factual info stabilizes, hallucinations oscillate
- **Formula**: std(alignments[-3:]) (lower is more stable)

#### 6. **Alignment Gain** (last - first layer)
- **What it measures**: Improvement from start to finish
- **Intuition**: How much did the representation evolve?
- **Factual**: Usually positive gain, hallucinations may have negative

#### 7. **Mean Velocity** (how fast representations change)
- **What it measures**: Rate of change between layers
- **Formula**: average of ||hidden_layer_{i+1} - hidden_layer_i||
- **Intuition**: Hallucinations often show erratic velocity changes

#### 8. **Mean Acceleration** (direction consistency)
- **What it measures**: Are changes consistent or chaotic?
- **Formula**: cosine similarity between consecutive deltas
- **Pattern**: Factual: smooth acceleration, Hallucination: chaotic

#### 9. **Oscillation Count** (direction changes)
- **What it measures**: How many times does alignment direction flip?
- **Intuition**: Hallucinations oscillate more (uncertainty)
- **Example**: Increasing, then decreasing, then increasing = 2 oscillations

## 4.2 Visualizing the Differences

```
FACTUAL STATEMENT TRAJECTORY:
Alignment
   1.0 |                    ~~~~~~ (stable plateau)
   0.8 |                 ~~~
   0.6 |              ~~~
   0.4 |           ~~~
   0.2 |        ~~~
   0.0 |_____~~~__________________
        L1  L2  L3  L4  L5  L6  L7  Layers
        (Early convergence, stable)

HALLUCINATION TRAJECTORY:
Alignment
   1.0 |
   0.8 |           ~~~~ (unstable peak)
   0.6 |        ~~~   ~~~
   0.4 |     ~~~         ~~~
   0.2 |  ~~~               ~~~
   0.0 |_~~~___________________~~~
        L1  L2  L3  L4  L5  L6  L7  Layers
        (Late peaks, oscillations)
```

---

# PART 5: TRAINING PROCESS - Step by Step

## 5.1 Data Preparation

### Step 1: Collect Pairs
```python
pairs = [
    ("Earth orbits Sun", "Earth revolves around Sun", "factual"),
    ("Earth orbits Moon", "Earth revolves around Sun", "hallucination"),
    # ... more pairs
]
```

### Step 2: Balance Classes
```python
# Count samples in each class
factuals = [p for p in pairs if p[2] == "factual"]
hallucinations = [p for p in pairs if p[2] == "hallucination"]

# Take equal number from each
min_count = min(len(factuals), len(hallucinations))
balanced = factuals[:min_count] + hallucinations[:min_count]
random.shuffle(balanced)
```

### Step 3: Create DataLoader
```python
dataset = TextPairDataset(balanced)
dataloader = DataLoader(
    dataset, 
    batch_size=8,  # Process 8 pairs at once
    shuffle=True   # Randomize order each epoch
)
```

## 5.2 Forward Pass - What Happens in One Batch

### For each batch of 8 pairs:

1. **Extract texts and truths**:
```python
texts = ["Earth orbits Sun", "Earth orbits Moon", ...]
truths = ["Earth revolves around Sun", "Earth revolves around Sun", ...]
labels = [1, 0, ...]  # 1 = factual, 0 = hallucination
```

2. **Get hidden states from LLM**:
```python
hidden_states = extractor.get_hidden_states(texts)
# Shape: [8, 12, 768]  (batch=8, layers=12, dim=768)
last_layer = hidden_states[:, -1, :]  # Take last layer
# Shape: [8, 768]
```

3. **Get truth embeddings**:
```python
truth_embeddings = truth_encoder.encode_batch(truths)
# Shape: [8, 384]  (MiniLM dimension)
```

4. **Project to shared space**:
```python
hidden_projected = hidden_proj(last_layer)        # [8, 256]
truth_projected = truth_proj(truth_embeddings)    # [8, 256]

# Normalize (important for cosine similarity)
hidden_projected = F.normalize(hidden_projected, p=2, dim=-1)
truth_projected = F.normalize(truth_projected, p=2, dim=-1)
```

5. **Compute similarities**:
```python
cos_sim = F.cosine_similarity(hidden_projected, truth_projected, dim=-1)
# Returns: [0.9, -0.3, 0.8, 0.2, ...]  # 8 similarity scores
```

6. **Calculate loss**:
```python
loss = enhanced_contrastive_loss(cos_sim, labels)
# Returns single number: e.g., 0.234
```

## 5.3 Backward Pass - Learning from Mistakes

### How Gradient Descent Works

Think of training like hiking down a mountain in fog:
- **Loss** = how high you are (want to go down)
- **Gradients** = which direction is downhill
- **Learning rate** = how big a step to take

```python
# Step 1: Calculate gradients
loss.backward()  # Computes how much each weight contributed to error

# Step 2: Clip gradients (prevent explosion)
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# Step 3: Update weights
optimizer.step()  # Move weights in direction that reduces loss

# Step 4: Adjust learning rate
scheduler.step()  # Gradually reduce learning rate over time
```

### Visualizing Weight Updates

```
BEFORE UPDATE:
Weight = 0.5
Gradient = +0.1 (increasing weight increases loss)
Learning rate = 0.01
New weight = 0.5 - 0.01*0.1 = 0.499

AFTER UPDATE:
Weight = 0.499 (slightly smaller)
Loss should be slightly lower
```

## 5.4 Monitoring Training Progress

### Key Metrics to Watch:

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease (if it increases, overfitting)
3. **Positive Similarity**: Should increase (factual pairs get closer)
4. **Negative Similarity**: Should decrease (hallucinations pushed apart)
5. **Validation Accuracy**: Should improve

### Example Training Log:
```
Epoch 1: loss=0.856, pos_sim=0.32, neg_sim=0.28, val_acc=0.52
Epoch 2: loss=0.621, pos_sim=0.48, neg_sim=0.15, val_acc=0.63
Epoch 3: loss=0.453, pos_sim=0.62, neg_sim=0.02, val_acc=0.71
Epoch 4: loss=0.321, pos_sim=0.74, neg_sim=-0.12, val_acc=0.78
Epoch 5: loss=0.234, pos_sim=0.83, neg_sim=-0.24, val_acc=0.84
```

---

# PART 6: EVALUATION METHODS

## 6.1 Supervised Evaluation

### The Process

1. **Extract features** from all samples (the 9 trajectory features)
2. **Split data** into training and testing sets (80/20)
3. **Train classifiers** on the features
4. **Evaluate** on test set

### Classifiers We Use:

#### Logistic Regression
- Simple but interpretable
- Learns linear decision boundaries
- Great baseline

#### Random Forest
- Ensemble of decision trees
- Captures non-linear patterns
- Robust to outliers

#### Gradient Boosting
- Sequential improvements
- Often best performance
- More prone to overfitting

### Evaluation Metrics Explained

#### Confusion Matrix Terms:
```
Actual\Predicted | Factual | Hallucination
-----------------|---------|---------------
Factual          |    TP   |      FN
Hallucination    |    FP   |      TN

TP = True Positives (correctly identified factual)
TN = True Negatives (correctly identified hallucination)
FP = False Positives (hallucination labeled as factual)
FN = False Negatives (factual labeled as hallucination)
```

#### Key Metrics:

1. **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness
   - Problematic if classes are imbalanced

2. **Precision** = TP / (TP + FP)
   - When we say "factual", how often are we right?
   - High precision = few false alarms

3. **Recall** = TP / (TP + FN)
   - What proportion of factuals did we catch?
   - High recall = few missed factuals

4. **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   - Balanced measure

5. **Specificity** = TN / (TN + FP)
   - What proportion of hallucinations did we catch?
   - Important for safety-critical applications

6. **AUC-ROC** (Area Under ROC Curve)
   - Measures ability to distinguish between classes
   - 1.0 = perfect, 0.5 = random
   - Independent of classification threshold

7. **MCC** (Matthews Correlation Coefficient)
   - Correlation between predictions and actual
   - Range: -1 to 1 (1 = perfect, 0 = random)
   - Balanced even with class imbalance

## 6.2 Unsupervised Evaluation

### Why Unsupervised?
Sometimes we don't have labeled data. We can still find patterns!

### Clustering Approach

```python
# Extract features
X = df[feature_columns].values

# Scale features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering (k=2 for factual vs hallucination)
kmeans = KMeans(n_clusters=2)
cluster_labels = kmeans.fit_predict(X_scaled)

# Compare with true labels (if available)
# Need to try both cluster mappings
accuracy1 = accuracy_score(true_labels, cluster_labels)
accuracy2 = accuracy_score(true_labels, 1 - cluster_labels)
clustering_accuracy = max(accuracy1, accuracy2)
```

### Anomaly Detection

```python
# Gaussian Mixture Model
gmm = GaussianMixture(n_components=2)
gmm.fit(X_scaled)

# Get anomaly scores (lower = more anomalous)
anomaly_scores = gmm.score_samples(X_scaled)

# Factual samples should have higher scores (more "normal")
# Hallucinations should have lower scores (anomalies)
```

## 6.3 Hybrid Evaluation

Combines both approaches:

```python
hybrid_score = 0.7 * best_supervised_score + 0.3 * clustering_accuracy
```

---

# PART 7: STATISTICAL ANALYSIS

## 7.1 T-Tests - Are Differences Significant?

### What is a T-test?
Determines if the difference between two groups is statistically significant.

### Example: Comparing final_alignment

```python
factual_alignments = [0.92, 0.88, 0.95, 0.89, 0.91]
hallucination_alignments = [0.23, 0.31, 0.18, 0.42, 0.15]

t_stat, p_value = ttest_ind(factual_alignments, hallucination_alignments)
```

### Interpreting P-values:
- **p < 0.05**: Statistically significant (95% confidence)
- **p < 0.01**: Highly significant (99% confidence)
- **p > 0.05**: Not significant (difference could be by chance)

## 7.2 Effect Size - How Big is the Difference?

### Cohen's d
Measures the magnitude of difference, independent of sample size.

```python
def cohens_d(group1, group2):
    diff = mean(group1) - mean(group2)
    pooled_std = sqrt((std1^2 + std2^2) / 2)
    return diff / pooled_std
```

### Interpreting Cohen's d:
- **d = 0.2**: Small effect
- **d = 0.5**: Medium effect  
- **d = 0.8**: Large effect

### Example Interpretation:
```
final_alignment:
  - p-value: 0.000001 (highly significant)
  - Cohen's d: 2.3 (very large effect)
  - Conclusion: Strong evidence factual and hallucination differ
```

---

# PART 8: PRACTICAL USAGE GUIDE

## 8.1 Installation and Setup

```bash
# Clone or create project
mkdir layerwise-semantic-dynamics
cd layerwise-semantic-dynamics

# Install dependencies
pip install torch transformers sentence-transformers
pip install datasets scikit-learn pandas numpy
pip install matplotlib seaborn scipy tqdm
```

## 8.2 Quick Start - Running Your First Analysis

```python
# minimal_example.py
from layerwise_semantic_dynamics import (
    LayerwiseSemanticDynamicsConfig,
    AnalysisOrchestrator,
    OperationMode
)

# Create configuration
config = LayerwiseSemanticDynamicsConfig(
    model_name="gpt2",           # Which LLM to analyze
    num_pairs=500,                # How many examples to use
    epochs=10,                    # Training epochs
    mode=OperationMode.HYBRID     # Evaluation mode
)

# Run analysis
orchestrator = AnalysisOrchestrator(config)
results = orchestrator.run_comprehensive_analysis()

# Check results
print(f"Best F1 Score: {results['key_findings']['best_f1_score']:.3f}")
print(f"Detection Quality: {results['key_findings']['detection_quality']}")
```

## 8.3 Configuration Options Explained

```python
config = LayerwiseSemanticDynamicsConfig(
    # Model settings
    model_name="gpt2",  # or "gpt2-medium", "facebook/opt-125m", etc.
    truth_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    
    # Training parameters
    batch_size=8,          # Higher = faster but more memory
    epochs=30,             # More epochs = better until overfitting
    learning_rate=5e-5,    # Too high = unstable, too low = slow
    margin=0.5,            # Separation required for negative pairs
    
    # Data settings
    num_pairs=1000,        # Total examples to use
    datasets=["synthetic", "truthfulqa"],  # Data sources
    
    # Operation mode
    mode=OperationMode.HYBRID,  # SUPERVISED, UNSUPERVISED, or HYBRID
    use_pretrained=False,       # Use previously trained models
    
    # Evaluation
    metrics=['f1', 'auroc', 'precision', 'recall'],
    confidence_threshold=0.7,    # Threshold for classification
)
```

## 8.4 Understanding the Output

### Directory Structure After Running:
```
layerwise_semantic_dynamics_system/
├── models/
│   ├── hidden_proj_best.pt     # Best hidden projector
│   ├── truth_proj_best.pt      # Best truth projector
│   └── hidden_proj_final.pt    # Final models
├── plots/
│   └── comprehensive_metrics.png  # Visualization
├── results/
│   ├── final_analysis_results.csv  # Raw features
│   ├── evaluation_results.json     # Metrics
│   ├── statistical_summary.json    # Stats tests
│   └── final_report.json           # Complete report
├── data/                          # Cached data
├── cache/                         # Model cache
└── execution.log                  # Detailed logs
```

### Key Files to Examine:

1. **final_report.json** - Start here for summary
2. **comprehensive_metrics.png** - Visual overview
3. **evaluation_results.json** - Detailed metrics
4. **execution.log** - Training progress and errors

## 8.5 Interpreting Results

### Example Good Results:
```json
{
  "key_findings": {
    "best_composite_score": 0.94,
    "best_f1_score": 0.92,
    "detection_quality": "EXCELLENT",
    "significant_metrics": 7
  },
  "recommendations": [
    "✓ Ready for production deployment in critical applications",
    "✓ Strong statistical foundation with multiple significant metrics",
    "✓ Good sample size for reliable analysis"
  ]
}
```

### Example Concerning Results:
```json
{
  "key_findings": {
    "best_composite_score": 0.68,
    "best_f1_score": 0.65,
    "detection_quality": "MODERATE",
    "significant_metrics": 2
  },
  "recommendations": [
    "⚠ Consider further optimization before production deployment",
    "⚠ Limited statistical significance - consider more data",
    "⚠ Small sample size - collect more data for robust results"
  ]
}
```

---

# PART 9: ADVANCED TOPICS

## 9.1 Customizing for Your Own Data

### Adding Custom Datasets

```python
# 1. Prepare your data in the right format
my_pairs = [
    ("Your text here", "Ground truth here", "factual"),
    ("Another text", "Different truth", "hallucination"),
    # ... more pairs
]

# 2. Extend DataManager or create custom loader
class CustomDataManager(DataManager):
    def load_my_dataset(self):
        # Load your data from files, database, etc.
        return my_pairs

# 3. Use in configuration
config.datasets = ["synthetic", "truthfulqa", "custom"]
```

### Data Format Requirements:
- **Text**: The model output you want to analyze
- **Truth**: The factual ground truth
- **Label**: "factual" or "hallucination"

## 9.2 Fine-tuning on Domain-Specific Data

### Medical Domain Example:
```python
medical_pairs = [
    ("Aspirin reduces fever", "Aspirin is an antipyretic medication", "factual"),
    ("Aspirin cures cancer", "Aspirin is not a cancer treatment", "hallucination"),
    # ... more medical pairs
]

# Train on mixed data
all_pairs = synthetic_pairs + medical_pairs
```

## 9.3 Ensemble Methods for Better Performance

### Combining Multiple Models:
```python
# Train multiple models with different seeds
configs = [
    LayerwiseSemanticDynamicsConfig(seed=42),
    LayerwiseSemanticDynamicsConfig(seed=123),
    LayerwiseSemanticDynamicsConfig(seed=456)
]

# Ensemble predictions
all_predictions = []
for config in configs:
    orchestrator = AnalysisOrchestrator(config)
    results = orchestrator.run_comprehensive_analysis()
    predictions = results['predictions']
    all_predictions.append(predictions)

# Average or vote
final_prediction = np.mean(all_predictions, axis=0)
```

## 9.4 Real-time Detection Integration

### API Example:
```python
class HallucinationDetector:
    def __init__(self, model_path="pretrained_models"):
        self.model_manager = ModelManager.load(model_path)
        self.feature_extractor = FeatureExtractor(self.model_manager)
    
    def detect(self, text, ground_truth):
        # Extract features
        features = self.feature_extractor.extract_trajectory_features(
            text, ground_truth
        )
        
        # Classify
        is_factual = self.classifier.predict([features])[0]
        confidence = self.classifier.predict_proba([features])[0]
        
        return {
            'is_factual': bool(is_factual),
            'confidence': float(confidence),
            'features': features
        }

# Usage
detector = HallucinationDetector()
result = detector.detect(
    "The Eiffel Tower is in Rome",
    "The Eiffel Tower is in Paris"
)
print(f"Hallucination detected: {not result['is_factual']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 9.5 Troubleshooting Common Issues

### Issue 1: Poor Performance
```
Symptoms: Low accuracy, random guessing
Solutions:
- Increase training data (num_pairs)
- Adjust learning rate (try 1e-4 or 1e-5)
- Increase epochs (watch for overfitting)
- Check class balance
```

### Issue 2: Overfitting
```
Symptoms: Train loss decreases, val loss increases
Solutions:
- Increase dropout (0.2 → 0.3)
- Add weight decay
- Reduce model complexity
- Get more training data
```

### Issue 3: Slow Training
```
Symptoms: Hours per epoch
Solutions:
- Reduce batch size (if memory limited)
- Use smaller model (gpt2 instead of gpt2-large)
- Cache hidden states (use_cache=True)
- Reduce max_length (if texts are short)
```

### Issue 4: Out of Memory
```
Symptoms: CUDA out of memory error
Solutions:
- Reduce batch size (8 → 4 → 2 → 1)
- Use gradient accumulation
- Use CPU if GPU memory insufficient
- Reduce max_length
```

---

# PART 10: THEORY DEEP DIVE

## 10.1 Why Does This Work? The Theoretical Foundation

### Information Flow in Neural Networks

Think of each layer as a transformation:
```
Layer 1: f₁(x) = σ(W₁x + b₁)
Layer 2: f₂(h₁) = σ(W₂h₁ + b₂)
...
Layer L: f_L(h_{L-1}) = σ(W_L h_{L-1} + b_L)
```

Where:
- x = input embedding
- h_i = hidden state at layer i
- W_i = weights (learned parameters)
- b_i = biases
- σ = activation function (non-linearity)

### The Manifold Hypothesis

Neural networks learn to map inputs onto a **manifold** (a lower-dimensional surface) in the high-dimensional space. Factual information lies on a different manifold than hallucinations.

```
High-dimensional space (e.g., 768 dimensions)
    │
    ├── Factual Manifold (smooth, continuous)
    │   ├── "Earth orbits Sun"
    │   ├── "Earth revolves around Sun"
    │   └── "The Earth travels around the Sun"
    │
    └── Hallucination Manifold (disconnected, sparse)
        ├── "Earth orbits Moon"
        ├── "Earth revolves around Mars"
        └── "The Earth is flat"
```

### Trajectory Through Layers

As information flows through layers, it moves along this manifold. The trajectory properties differ:

**Factual Trajectory**:
- Moves toward stable attractor states
- Converges quickly
- Shows smooth, monotonic improvement

**Hallucination Trajectory**:
- Wanders in unstable regions
- May approach then leave attractors
- Shows oscillatory behavior

## 10.2 Mathematical Formulation

### Layer-wise Alignment:
For layer l, alignment a_l is:
```
a_l = cos(proj_hidden(h_l), proj_truth(t))
     = (proj_hidden(h_l) · proj_truth(t)) / (||proj_hidden(h_l)|| ||proj_truth(t)||)
```

### Trajectory Feature Calculation:

**Stability** (variance in last k layers):
```
S = Var(a_{L-k}, a_{L-k+1}, ..., a_L)
```

**Velocity** (rate of change):
```
v_l = ||h_{l+1} - h_l||_2
```

**Acceleration** (consistency of change):
```
α_l = cos(v_{l+1}, v_l)
```

**Oscillation Count**:
```
O = #{l | sign(a_{l+1} - a_l) ≠ sign(a_l - a_{l-1})}
```

## 10.3 Connection to Model Confidence

There's a strong correlation between trajectory features and model confidence:

```
High Confidence → Early convergence, high stability
Low Confidence → Late convergence, oscillations
Hallucination → Pathological trajectories (unstable, oscillatory)
```

This suggests the model "knows" when it's hallucinating at intermediate layers, even if the final output is confident.

---

# PART 11: APPLICATIONS AND USE CASES

## 11.1 Content Moderation

**Use Case**: Detect AI-generated misinformation

```python
def moderate_content(text, source_claims):
    # Extract claims from text
    claims = extract_claims(text)
    
    # Check each claim against source
    suspicious_claims = []
    for claim in claims:
        for source_claim in source_claims:
            result = detector.detect(claim, source_claim)
            if not result['is_factual'] and result['confidence'] > 0.8:
                suspicious_claims.append(claim)
    
    return suspicious_claims
```

## 11.2 RAG System Validation

**Use Case**: Verify retrieval-augmented generation outputs

```python
class RAGValidator:
    def __init__(self, detector):
        self.detector = detector
    
    def validate_response(self, query, response, retrieved_docs):
        # Check response against each retrieved document
        scores = []
        for doc in retrieved_docs:
            result = self.detector.detect(response, doc)
            scores.append(result['confidence'])
        
        # Average confidence across relevant docs
        confidence = np.mean(scores)
        
        return {
            'is_valid': confidence > 0.7,
            'confidence': confidence,
            'per_doc_scores': scores
        }
```

## 11.3 Model Comparison

**Use Case**: Compare different models' tendency to hallucinate

```python
models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
hallucination_rates = []

for model in models:
    config.model_name = model
    orchestrator = AnalysisOrchestrator(config)
    results = orchestrator.run_comprehensive_analysis()
    
    # Extract hallucination rate from results
    rate = 1 - results['dataset_statistics']['class_balance']
    hallucination_rates.append(rate)

# Plot comparison
plt.bar(models, hallucination_rates)
plt.title("Hallucination Rate by Model Size")
plt.show()
```

## 11.4 Training Data Quality Assessment

**Use Case**: Identify potential hallucinations in training data

```python
def audit_training_data(dataset, detector):
    issues = []
    
    for i, example in enumerate(dataset):
        text = example['text']
        target = example['target']
        
        result = detector.detect(text, target)
        if not result['is_factual'] and result['confidence'] > 0.9:
            issues.append({
                'index': i,
                'text': text,
                'target': target,
                'confidence': result['confidence']
            })
    
    return issues
```

---

# PART 12: PERFORMANCE OPTIMIZATION

## 12.1 Speed Optimizations

### Caching Hidden States
```python
# Enable caching
extractor = HiddenStatesExtractor(
    model_name="gpt2",
    use_cache=True  # Cache repeated texts
)

# Clear cache when done
extractor.clear_cache()
```

### Batch Processing
```python
# Process in batches for efficiency
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_truths = truths[i:i+batch_size]
    features = extract_batch_features(batch_texts, batch_truths)
```

### Mixed Precision Training
```python
# Use FP16 for faster training
with torch.cuda.amp.autocast():
    hidden_states = extractor.get_hidden_states(texts)
    loss = compute_loss(hidden_states, truths, labels)
```

## 12.2 Memory Optimizations

### Gradient Accumulation
```python
# Simulate larger batch with limited memory
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Layer-wise Processing
```python
# Process one layer at a time to save memory
all_alignments = []
for layer_idx in range(num_layers):
    layer_hidden = hidden_states[:, layer_idx, :]
    alignment = compute_alignment(layer_hidden, truth_embedding)
    all_alignments.append(alignment)
```

## 12.3 Accuracy Optimizations

### Ensemble Methods
```python
# Combine multiple models
classifiers = [
    LogisticRegression(),
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier()
]

predictions = []
for clf in classifiers:
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    predictions.append(pred)

# Average predictions
ensemble_pred = np.mean(predictions, axis=0)
```

### Feature Selection
```python
# Identify most important features
selector = SelectKBest(k=5)  # Keep top 5 features
X_selected = selector.fit_transform(X, y)

# Which features were selected?
selected_features = feature_columns[selector.get_support()]
print(f"Most important: {selected_features}")
```

---

# PART 13: EXTENDING THE SYSTEM

## 13.1 Adding New Features

```python
class EnhancedFeatureExtractor(FeatureExtractor):
    def extract_additional_features(self, text, truth):
        base_features = super().extract_trajectory_features(text, truth)
        
        # Add entropy-based features
        entropy_features = self.compute_entropy_features(text, truth)
        
        # Add attention-based features (if available)
        attention_features = self.compute_attention_features(text)
        
        return {**base_features, **entropy_features, **attention_features}
    
    def compute_entropy_features(self, text, truth):
        # Calculate prediction entropy across layers
        # Higher entropy = more uncertainty
        return {
            'final_entropy': entropy,
            'entropy_trajectory': entropy_trajectory
        }
```

## 13.2 Adding New Evaluation Metrics

```python
class ExtendedEvaluator(ComprehensiveEvaluator):
    def compute_additional_metrics(self, y_true, y_pred, y_pred_proba):
        base_metrics = super()._compute_comprehensive_metrics(
            y_true, y_pred, y_pred_proba
        )
        
        # Add Brier score
        brier = np.mean((y_pred_proba - y_true) ** 2)
        
        # Add log loss
        log_loss = -np.mean(
            y_true * np.log(y_pred_proba + 1e-15) + 
            (1 - y_true) * np.log(1 - y_pred_proba + 1e-15)
        )
        
        base_metrics.update({
            'brier_score': float(brier),
            'log_loss': float(log_loss)
        })
        
        return base_metrics
```

## 13.3 Adding Visualization Types

```python
class ExtendedVisualizer(VisualizationEngine):
    def plot_layer_heatmap(self, layerwise_data):
        """Plot heatmap of alignments across layers and samples"""
        
        # Prepare data
        factual_trajs = np.array(layerwise_data['factual'])
        hallucination_trajs = np.array(layerwise_data['hallucination'])
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(factual_trajs, ax=ax1, cmap='RdYlGn', 
                   cbar_kws={'label': 'Alignment'})
        ax1.set_title('Factual Samples Trajectories')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Sample')
        
        sns.heatmap(hallucination_trajs, ax=ax2, cmap='RdYlGn',
                   cbar_kws={'label': 'Alignment'})
        ax2.set_title('Hallucination Samples Trajectories')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Sample')
        
        plt.tight_layout()
        plt.savefig(dir_manager.get_plot_path("layer_heatmap"))
        plt.show()
```

---

# PART 14: DEPLOYMENT GUIDE

## 14.1 Model Export

```python
# Save trained models in deployment format
def export_for_deployment(model_manager, output_dir):
    # Save projection heads
    torch.save(
        model_manager.hidden_proj.state_dict(),
        f"{output_dir}/hidden_proj.pt"
    )
    torch.save(
        model_manager.truth_proj.state_dict(),
        f"{output_dir}/truth_proj.pt"
    )
    
    # Save feature scaler
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    
    # Save classifier
    joblib.dump(classifier, f"{output_dir}/classifier.pkl")
    
    # Save config
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config.__dict__, f)
```

## 14.2 REST API Example

```python
# app.py
from flask import Flask, request, jsonify
import torch
import joblib

app = Flask(__name__)

# Load models at startup
classifier = joblib.load("models/classifier.pkl")
scaler = joblib.load("models/scaler.pkl")
model_manager = ModelManager.load("models/")

@app.route('/detect', methods=['POST'])
def detect_hallucination():
    data = request.json
    text = data['text']
    truth = data['truth']
    
    # Extract features
    features = feature_extractor.extract_trajectory_features(text, truth)
    
    # Scale and predict
    features_scaled = scaler.transform([list(features.values())])
    prediction = classifier.predict(features_scaled)[0]
    confidence = classifier.predict_proba(features_scaled)[0].max()
    
    return jsonify({
        'is_factual': bool(prediction),
        'confidence': float(confidence),
        'features': features
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 14.3 Performance Monitoring

```python
class DeploymentMonitor:
    def __init__(self):
        self.predictions = []
        self.true_labels = []
        self.latencies = []
    
    def log_prediction(self, text, truth, prediction, confidence, latency):
        self.predictions.append({
            'text': text,
            'truth': truth,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'latency_ms': latency * 1000
        })
    
    def get_stats(self):
        if not self.predictions:
            return {}
        
        df = pd.DataFrame(self.predictions)
        
        return {
            'total_queries': len(df),
            'avg_confidence': df['confidence'].mean(),
            'avg_latency_ms': df['latency_ms'].mean(),
            'p95_latency_ms': df['latency_ms'].quantile(0.95),
            'factual_rate': df['prediction'].mean()
        }
    
    def detect_drift(self, window_size=100):
        """Detect if model performance is drifting"""
        recent = self.predictions[-window_size:]
        historical = self.predictions[:-window_size]
        
        if len(recent) < window_size or len(historical) < window_size:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        # Compare distributions
        recent_conf = [p['confidence'] for p in recent]
        hist_conf = [p['confidence'] for p in historical]
        
        t_stat, p_value = ttest_ind(recent_conf, hist_conf)
        
        return {
            'drift_detected': p_value < 0.05,
            'p_value': p_value,
            'recent_mean': np.mean(recent_conf),
            'historical_mean': np.mean(hist_conf)
        }
```

---

# PART 15: TROUBLESHOOTING AND FAQ

## 15.1 Common Issues and Solutions

### Q1: Why is my accuracy stuck at 50%?
**A**: This suggests random guessing. Check:
- Data balance (should be ~50/50 factual/hallucination)
- Learning rate (too high or too low)
- Model capacity (projection heads too small)
- Training epochs (need more iterations)

### Q2: Why does validation loss increase while training loss decreases?
**A**: This is overfitting. Solutions:
- Increase dropout (0.2 → 0.3 or 0.4)
- Add weight decay (1e-4 → 1e-3)
- Reduce model complexity (smaller hidden dims)
- Get more training data

### Q3: Why are all predictions the same class?
**A**: Class imbalance or model collapse. Check:
- Class distribution in training data
- Loss function weights (try weighted loss)
- Threshold calibration

### Q4: Why is training so slow?
**A**: Several possibilities:
- Batch size too large (reduce)
- Model too large (try smaller variant)
- max_length too high (reduce if texts are short)
- Not using caching (enable use_cache=True)

### Q5: Why do I get CUDA out of memory?
**A**: Memory issues:
- Reduce batch size
- Use gradient accumulation
- Process in CPU mode
- Use model parallelism if available

## 15.2 Debugging Tips

### Enable Detailed Logging
```python
logger.level = "DEBUG"  # See all messages

# Add custom debug points
def debug_tensor(name, tensor):
    print(f"{name}: shape={tensor.shape}, "
          f"mean={tensor.mean():.3f}, "
          f"std={tensor.std():.3f}")
```

### Check Gradient Flow
```python
# Monitor gradients during training
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

### Validate Data Pipeline
```python
# Test with single batch
sample_batch = next(iter(train_loader))
texts = [item['text'] for item in sample_batch]
truths = [item['truth'] for item in sample_batch]
labels = torch.tensor([item['label'] for item in sample_batch])

# Forward pass
hidden_states = extractor.get_hidden_states(texts)
truth_emb = truth_encoder.encode_batch(truths)
print(f"Hidden states shape: {hidden_states.shape}")
print(f"Truth embeddings shape: {truth_emb.shape}")
```

---

# APPENDIX: QUICK REFERENCE

## Key Parameters Summary

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| batch_size | 2-64 | 8 | Higher = faster but more memory |
| learning_rate | 1e-6 to 1e-3 | 5e-5 | Higher = faster but unstable |
| epochs | 5-100 | 30 | More = better until overfitting |
| margin | 0.1-1.0 | 0.5 | Higher = stricter separation |
| dropout | 0.0-0.5 | 0.2 | Higher = more regularization |
| weight_decay | 0-1e-2 | 1e-5 | Higher = more regularization |

## Feature Reference

| Feature | Range | Interpretation |
|---------|-------|----------------|
| final_alignment | [-1, 1] | Higher = more factual |
| mean_alignment | [-1, 1] | Higher = more factual |
| max_alignment | [-1, 1] | Higher = more factual |
| convergence_layer | [0, L-1] | Earlier = faster convergence |
| stability | [0, 2] | Lower = more stable |
| alignment_gain | [-2, 2] | Positive = improvement |
| mean_velocity | [0, ∞) | Lower = smoother changes |
| mean_acceleration | [-1, 1] | Higher = more consistent |
| oscillation_count | [0, L-2] | Lower = more stable |

## Metric Reference

| Metric | Range | Perfect | Random |
|--------|-------|---------|--------|
| Accuracy | [0, 1] | 1.0 | 0.5 |
| Precision | [0, 1] | 1.0 | 0.5 |
| Recall | [0, 1] | 1.0 | 0.5 |
| F1 | [0, 1] | 1.0 | 0.5 |
| AUC-ROC | [0, 1] | 1.0 | 0.5 |
| MCC | [-1, 1] | 1.0 | 0.0 |
| Cohen's Kappa | [-1, 1] | 1.0 | 0.0 |

---

# CONCLUSION

Layer-wise Semantic Dynamics provides a powerful framework for detecting hallucinations in language models by analyzing how representations evolve through the network. This approach works because:

1. **Factual information follows stable trajectories** through the layers, converging early to semantically meaningful representations

2. **Hallucinations show pathological trajectories** - oscillating, converging late or not at all, and showing instability

3. **The shared space projection** enables direct comparison between model outputs and ground truth, even when they use different vocabularies

4. **Comprehensive feature extraction** captures multiple aspects of these trajectories, providing rich signals for classification

5. **Multiple evaluation strategies** (supervised, unsupervised, hybrid) provide robustness and adaptability to different use cases

The system is designed to be:
- **Extensible** - add new features, models, or datasets
- **Scalable** - from single examples to large batches
- **Interpretable** - understand why decisions are made
- **Practical** - ready for real-world deployment

Whether you're building content moderation systems, validating RAG pipelines, or researching model behavior, this framework provides the tools you need to detect and understand hallucinations in language models.