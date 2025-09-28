# 🌳 LogicLeaf: Decision Tree Visualizer and Naive Bayes Classifier

A Python-based implementation of **Decision Trees** built from scratch, (complete with **Entropy (Information Gain)** and **Gini Index** as splitting criteria) and **Naive Bayes
Classifier**. The project also includes a **Graphviz-powered visualizer** to generate crisp, interpretable tree diagrams.  


## 🚀 Features  
- **Decision Tree Implementation**
  - **Custom Entropy & Gini Functions** – Implemented from scratch, no external ML libraries.  
  - **Dynamic Tree Builder** – Recursively constructs decision trees using chosen impurity measures.  
  - **Dual Criteria Support**  
    - *Entropy (Information Gain)* → ID3-style splitting  
    - *Gini Index* → CART-style splitting  
  - **Interactive Visualizations** – Trees are exported as PNGs with Graphviz.  
  - **Human-readable structure** – Leaf nodes represent final decisions, internal nodes show feature splits.
-   **Naive Bayes Classifier**
    -   Probabilistic model using Bayes' theorem.
    -   **Laplace Smoothing Applied** – Prevents zero-probability issues for unseen feature values in training.
    -   **Supports Multi-class Classification** – Works seamlessly with more than two decision categories.
    -   **Randomized Testing** – Automatically selects a random instance from the dataset for prediction and evaluation.


## 📂 Project Structure  
```bash
DecisionTreeVisualizer/
│── data.csv             # Input Dataset
│── decision_tree.py     # Main script with tree logic + visualization
│── tree_entropy.png     # Tree built using entropy
│── tree_gini.png        # Tree built using gini index
│── naive_bayes.py       # Implemenatation of Naive Bayes Classifier
│── README.md            # You are here
```


## ⚙️ How It Works

### 🌳 Decision Tree

1. **Entropy & Gini Calculation**  
   - Computes uncertainty of class labels.  
   - Lower impurity ⇒ better split.  

2. **Attribute Selection**  
   - Recursively selects the best feature based on chosen metric (*Entropy* or *Gini*).  

3. **Tree Construction**  
   - Splits dataset into subsets by feature values.  
   - Continues until pure leaves or no features remain.  

4. **Visualization**  
   - Uses **Graphviz (Digraph)** to generate interpretable flowchart-like trees.

For more details, kindly refer to: [Decission Tree.pdf](https://drive.google.com/file/d/19IFB7xIIxHt8mHSAaEXqxUOkRZEa6-vU/view?usp=sharing)
  
### 🎲 Naive Bayes Classifier

1.  **Training Phase**
    -   Computes priors *P(Class)*.
    -   Computes likelihoods *P(Feature=Value \| Class)* with Laplace 
    Smoothing.
2.  **Prediction Phase**
    - For new instances: **$P(C_k \mid X) \propto P(C_k) \times \prod_{i=1}^n P(x_i \mid C_k)$**
    - **The Naive Assumption considered here is that all the events are completely INDEPENDENT of each other. Hence their intersection can be written as the product of hte individual conditional probabilities.**
    - Normalizes probabilities.
    - Chooses class with highest posterior.
3. **Verbose Mode**
    - Prints priors, conditional probabilities, posterior products,
normalization, and final prediction.

For more details kindly refer to: [Naive Bayes Classifier.pdf](https://drive.google.com/file/d/1dfo-JBO46zeUZpcNcGQoVnSNgU5CGwT9/view?usp=sharing)


## 🛠 Installation  

Install the required Python libraries:  

```bash
pip install pandas numpy graphviz
```

Also, install the Graphviz system package (needed for rendering images):

### Debian/Ubuntu
```bash
sudo apt-get update && sudo apt-get install -y graphviz
```

### macOS (Homebrew)
```bash
brew install graphviz
```

### Windows
Download from [Graphviz.org](https://graphviz.gitlab.io/download/) and add it to your PATH.


## 📊 Usage

Run the script with your dataset (weekend.csv as default):

```bash
python decision_tree.py
python naive_bayes.py
```

This will generate:

```bash
tree_entropy.png   # Decision tree using Information Gain
tree_gini.png      # Decision tree using Gini Index
```

Both trees will be saved in the working directory and usually open automatically.

The output of the Naive Bayes Classifier will be visible in the terminal. 

## 📊 Example Input Data

Here is an example of the input dataset used in the project:

| Weekend | Weather | Parents | Financial Condition | Decision     |
|---------|---------|---------|---------------------|--------------|
| W1      | Sunny   | Yes     | Rich                | Cinema       |
| W2      | Sunny   | No      | Rich                | Play Tennis  |
| W3      | Windy   | Yes     | Rich                | Cinema       |
| W4      | Rainy   | Yes     | Poor                | Cinema       |
| W5      | Rainy   | No      | Rich                | Stay in      |
| W6      | Rainy   | Yes     | Poor                | Cinema       |
| W7      | Windy   | No      | Poor                | Cinema       |
| W8      | Windy   | No      | Rich                | Shopping     |
| W9      | Windy   | Yes     | Rich                | Cinema       |
| W10     | Sunny   | No      | Rich                | Play Tennis  |


## 🔍 Example Output

### 🌳 Decision Tree (Entropy vs Gini)

### Decision Tree Visual Guide

- 🟦 **Feature Nodes** → Light-blue rounded boxes  
- 🟩 **Leaf Nodes** → Green ellipses (final decision)  

#### 📌 Entropy-Based Trees  
- Maximize information gain  
<div align="center">
  <img width="894" height="413" alt="Entropy Tree" src="https://github.com/user-attachments/assets/2fb095ba-5a25-457c-b449-bcb756c3da0c" />
</div>

#### 📌 Gini-Based Trees  
- Minimize class impurity  
<div align="center">
  <img width="535" height="413" alt="Gini Tree" src="https://github.com/user-attachments/assets/eed20ab2-cf13-4bf6-be7c-316271f04a21" />
</div>

------------------------------------------------------------------------
### 🎲 Naive Bayes Classifier

-   🔵 **Priors** (class probabilities):
    -   P(Cinema) = 0.6
    -   P(Play Tennis) = 0.2
    -   P(Stay in) = 0.1
    -   P(Shopping) = 0.1
-   🟠 **Conditional Probabilities** (sample):
    -   P(Weather = Sunny \| Play Tennis) = 0.6
    -   P(Parents = Yes \| Cinema) = 0.75
    -   P(Financial Condition = Rich \| Shopping) = 0.67
-   🟢 **Posterior Probabilities** (normalized):
    -   P(Cinema \| features) = 0.1566
    -   P(Play Tennis \| features) = 0.6344
    -   P(Stay in \| features) = 0.1044
    -   P(Shopping \| features) = 0.1044

✅ **Final Prediction:** *Play Tennis* (matches actual label).

For detailed sample output of the Naive Bayes Classifier, kindly refer to: [Naive Bayes Classifier.pdf](https://drive.google.com/file/d/1dfo-JBO46zeUZpcNcGQoVnSNgU5CGwT9/view?usp=sharing)

## 🎯 Conclusion

The **Decision Tree** and **Naive Bayes Classifier** demonstrate two
foundational approaches to classification:
- **Decision Trees** → Rule-based, interpretable, visual structures.
- **Naive Bayes** → Probabilistic, fast, and effective with categorical
data.

Together, they showcase how classical ML algorithms can be built from
scratch and applied to real-world data for learning and prediction.
