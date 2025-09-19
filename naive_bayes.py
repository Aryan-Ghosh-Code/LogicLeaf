import pandas as pd
import random
from collections import defaultdict

data = pd.read_csv("data.csv")

features = list(data.columns[1:-1])  # exclude Weekend ID column
target = "Decision"

def train_naive_bayes(df, features, target): #Training Functio
    model = {
        "priors": {},
        "likelihoods": {f: defaultdict(dict) for f in features}
    }
    
    total_samples = len(df)
    class_counts = df[target].value_counts()
    
    # Calculate priors: P(Class)
    for c, count in class_counts.items():
        model["priors"][c] = count / total_samples
    
    # Calculate likelihoods: P(Feature=Value | Class)
    for feature in features:
        for c in class_counts.index:
            subset = df[df[target] == c]
            value_counts = subset[feature].value_counts()
            total_in_class = len(subset)
            unique_values = df[feature].nunique()
            
            for value in df[feature].unique():
                model["likelihoods"][feature][value][c] = (
                    (value_counts.get(value, 0) + 1) / (total_in_class + unique_values)
                )
    return model

def predict_naive_bayes_verbose(model, features, instance):
    posteriors = {}
    print("\n--- Detailed Posterior Calculation ---")
    
    for c in model["priors"]:
        print(f"\nClass: {c}")
        likelihood_product = 1
        
        for feature in features:
            value = instance[feature]
            prob = model["likelihoods"][feature].get(value, {}).get(c, 1e-6)
            likelihood_product *= prob
            print(f"P({feature}={value} | {c}) = {prob}")
        
        prior = model["priors"][c]
        likelihood_prior = likelihood_product * prior
        posteriors[c] = likelihood_prior
        print('Now let us take the "Naive" part into consideration, i.e. intersection of all the features')
        print(f"P(features | class) × P(class) = {likelihood_prior}")

    print("\nAfter Normalization\n")
    total = sum(posteriors.values())
    for c in posteriors:
        print(posteriors[c], total)
        posteriors[c] /= total
        print(f"Posterior P({c} | features) = {posteriors[c]}")
    
    predicted_class = max(posteriors, key=posteriors.get)
    return predicted_class

model = train_naive_bayes(data, features, target)

print("\n--- Priors (P(Class)) ---")
for c, p in model["priors"].items():
    print(f"P({c}) = {p}")

print("\n--- Conditional Probabilities (P(Feature=Value | Class)) ---")
for feature in features:
    print(f"\nFeature: {feature}")
    for value in data[feature].unique():
        probs = model["likelihoods"][feature][value]
        for cls, prob in probs.items():
            print(f"P({feature}={value} | {cls}) = {prob}")

random_index = random.randint(0, len(data) - 1) #  Random Test Sample Selection 
test_instance_row = data.iloc[random_index]
test_instance = test_instance_row[features].to_dict()
true_decision = test_instance_row[target]

print("\n--- Random Test Instance ---")
print(f"Row index: {random_index}")
for feature, value in test_instance.items():
    print(f"  {feature}: {value}")
print(f"  True Decision: {true_decision}")

# ------------------ Prediction ------------------
prediction = predict_naive_bayes_verbose(model, features, test_instance)

print("\n--- Final Prediction ---")
for feature, value in test_instance.items():
    print(f"  {feature}: {value}")
print(f"Predicted Decision: {prediction}")
print(f"Actual Decision: {true_decision}")
