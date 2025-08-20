# DeciVision
A Python-powered decision tree builder and visualizer that explores attribute selection using Entropy (ID3 style) and Gini Index (CART style). Features include: custom metric calculation (using Information Gain and Gini Index), recursive tree construction, and Graphviz-based visualization for intuitive interpretation of decision boundaries.

🚀 Features

Build decision trees using Entropy (ID3-style) or Gini Impurity (CART-style).

Custom implementation of metric calculations (no black-box libraries).

Recursive tree construction for clean, interpretable results.

Graphviz-powered visualization for elegant branching diagrams.

Modular and extensible design to experiment with datasets and metrics.

📦 Installation

Clone this repository and install the dependencies:

git clone https://github.com/your-username/BranchCraft.git
cd BranchCraft
pip install -r requirements.txt

🛠️ Usage

Run the main script with your dataset:

python main.py


This will:

Calculate attribute selection measures (Entropy/Gini).

Construct a decision tree recursively.

Output a Graphviz .dot file and rendered visualization (.png/.pdf).

📊 Example Output

A decision tree built with Entropy might look like:

            [Feature X?]
            /          \
        yes/            \no
       Leaf A         [Feature Y?]
                      /          \
                  yes/            \no
                Leaf B           Leaf C


And with Graphviz, it becomes an elegant branching diagram.

🔍 Why BranchCraft?

Instead of hiding complexity, BranchCraft exposes the mathematical underpinnings of decision trees — showing exactly how Entropy and Gini drive the branching process. It’s not just a tool; it’s a learning companion for students, researchers, and developers exploring interpretable machine learning.