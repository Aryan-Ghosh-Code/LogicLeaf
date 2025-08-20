# 🌳 DeciVision: Decision Tree Visualizer  

A Python-based implementation of **Decision Trees** built from scratch, complete with **Entropy (Information Gain)** and **Gini Index** as splitting criteria. The project also includes a **Graphviz-powered visualizer** to generate crisp, interpretable tree diagrams.  


## 🚀 Features  
- **Custom Entropy & Gini Functions** – Implemented from scratch, no external ML libraries.  
- **Dynamic Tree Builder** – Recursively constructs decision trees using chosen impurity measures.  
- **Dual Criteria Support**  
  - *Entropy (Information Gain)* → ID3-style splitting  
  - *Gini Index* → CART-style splitting  
- **Interactive Visualizations** – Trees are exported as PNGs with Graphviz.  
- **Human-readable structure** – Leaf nodes represent final decisions, internal nodes show feature splits.  


## 📂 Project Structure  
```bash
DecisionTreeVisualizer/
│── weekend.csv          # Input dataset
│── decision_tree.py     # Main script with tree logic + visualization
│── tree_entropy.png     # Tree built using entropy
│── tree_gini.png        # Tree built using gini index
│── README.md            # You are here
```


## ⚙️ How It Works

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
```

This will generate:

```bash
tree_entropy.png   # Decision tree using Information Gain
tree_gini.png      # Decision tree using Gini Index
```

Both trees will be saved in the working directory and usually open automatically.

Sure! Here’s the input part formatted as a table that you can directly paste into your README:

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

- 🟦 **Feature Nodes** → Light-blue rounded boxes
- 🟩 **Leaf Nodes** → Green ellipses (final decision)

📌 **Entropy-based trees maximize information gain**  

<div align="center">
    <img width="894" height="413" alt="Entropy Tree" src="https://github.com/user-attachments/assets/2fb095ba-5a25-457c-b449-bcb756c3da0c" />
</div>


📌 **Gini-based trees minimize class impurity**

<div align="center">
    <img width="535" height="413" alt="Gini Tree" src="https://github.com/user-attachments/assets/eed20ab2-cf13-4bf6-be7c-316271f04a21" />
</div>

