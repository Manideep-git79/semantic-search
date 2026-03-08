from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def load_dataset():

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers","footers","quotes")
    )

    df = pd.DataFrame({
        "text": dataset.data,
        "target": dataset.target
    })

    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print("Total documents:", len(df))