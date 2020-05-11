from collections import Counter
import matplotlib.pyplot as plt
import pickle

def show_star_counts(filename="data/stars.pickle"):
    with open(filename, 'rb') as f:
        stars = pickle.load(f)

    star_counts = Counter(stars)
    print(sorted(star_counts.items()))
    plt.bar(range(1, 6), [sc[1] for sc in sorted(star_counts.items())])
    plt.title("Distribution of star ratings")
    plt.show()

if __name__ == "__main__":
    show_star_counts()
