# 02_Model_Architecture: Model Architecture and Tensor Derivation
This module abandons all black-box frameworks. Directly in Jupyter Notebook (`.ipynb`) with PyTorch, it derives the underlying core tensor flow logic of **Multi-Head Attention** for large models from scratch.

## ⚙️ Core Environment
- **Virtual Environment**: `torch_env` (strictly physically isolated)
- **Dependency List**: `requirements.txt` (includes core tensor computation libraries such as `torch`, `numpy`, `ipykernel`)

---

## 🧠 Core Concept 1: The Physical Dimensions of Tensors
In the pipeline of large models, the skeleton of tensors consists of the following core dimensions, which directly determine the model's computing power distribution and cognitive capability:

- **$B$ (Batch Size)**: Batch size. The "parallel isolation barrier" during computation, a container for maximizing GPU parallel computing power. Texts from different batches are completely independent of one another.
- **$T$ (Sequence Length)**: Sequence length (context window). Analogous to "how many people are present at a matchmaking event". The larger $T$ is, the longer context the model can remember. However, the attention matrix of size $T \times T$ leads to an $O(T^2)$ explosion in computational cost.
- **$C$ (Embed Size)**: Total embedding dimension. Comparable to "brain capacity" or "the set of comprehensive attributes of an individual" (e.g., $C=4096$ for Llama-3-8B). A larger $C$ allows the model to capture richer and more nuanced features for each token.
- **$H$ (Num Heads)**: Number of attention heads. Analogous to "the count of review panels with different specializations".
- **$D$ (Head Dim)**: Dimension per attention head. An inviolable fundamental formula for large models: $C = H \times D$. To avoid simplistic blended scoring, the total dimension $C$ is split so that different review panels conduct independent evaluations within their respective $D$-dimensional subspaces (covering grammar, logic, sentiment, and more).

---

## ⚖️ Core Concept 2: The Asymmetric Supply-Demand Philosophy of $Q, K, V$
Why do we not compute similarity by multiplying the input tensor $X$ with itself ($X \times X^T$), but instead map to $Q, K, V$ via three weight matrices ($W_Q, W_K, W_V$)?
Answer: **To break inherent physical symmetry and build a directional supply-demand matching mechanism.**

1. **$Q$ (Query / Seeker)**: The identity of the current token acting as the demand side. For the verb "eat", for example, its $Q$ vector broadcasts across the whole sequence: "I need a noun that is edible".
2. **$K$ (Key / Billboard)**: The identity of the current token acting as the evaluated party. For the noun "apple", for example, its $K$ vector exposes attribute labels: "I am fruit and edible".
3. **$V$ (Value / Core Asset)**: The genuine semantic features extracted from the current token.
**At the moment of dot-product calculation ($Q \times K^T$)**: Supply and demand align perfectly. Nodes with collinear features obtain high scores, forming an affinity matrix full of asymmetry and directionality.

---

## 🌀 Core Concept 3: The Spatial Magic of MHA (Split & Splicing)
Within the `forward` function of industrial-grade `MultiHeadAttention` implementations, tensors undergo an elegant reshaping transformation:

1. **Spatial Split**: $Q.view$ splits the final dimension $C$ into $H \times D$, changing the shape to $(B, T, H, D)$.
2. **Parallel Space**: $.transpose(1, 2)$ moves $H$ forward, reshaping the tensor to $(B, H, T, D)$. At this point, $B$ and $H$ together form isolation boundaries, creating $B \times H$ independent 2D matrix multiplication spaces at the underlying level.
3. **Feature Recombination**: After affinity calculation $(T, D) \times (D, T) \rightarrow (T, T)$ and value extraction within these parallel spaces, the tensor is transposed back.
4. **View Concatenation**: $.view(B, T, C)$ flattens the $H$ and $D$ dimensions. Local outputs from all independent attention heads are concatenated sequentially and seamlessly merged back into the final high-dimensional representation.

---

## 🏭 Extended Perspective: Macro Lifecycle of Large Models
- **Training**: Extensive computational resources are consumed. Backpropagation continuously fine-tunes floating-point values in thousands of $Q, K, V$ projection matrices and FFN (Feed-Forward Network, responsible for storing encyclopedic knowledge and memory) matrices inside the network.
- **Inference**: All matrices are frozen. The input prompt becomes the initial tensor $X_0$. It passes through dozens of fixed Attention and FFN layers, progressing from literal comprehension to deep logical abstraction, and finally maps to a probability distribution over the vocabulary to predict the next token.

## 🚀 Current Engineering Progress
- [x] Successfully implemented the matrix multiplication and Softmax stabilization normalization for single-head Self-Attention in the `.ipynb` experimental environment.
- [x] Used `.view` and `.transpose` to complete OOP encapsulation of industrial-grade **Multi-Head Attention**, and verified strict consistency of input and output with dimension shape $(B, T, C)$.