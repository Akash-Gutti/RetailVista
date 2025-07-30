import os
import random
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)


def simulate_pos_data(n=10000, output_path="data/raw/pos_data.csv"):
    skus = [f"SKU_{i}" for i in range(100, 200)]
    stores = [f"Store_{i}" for i in range(1, 11)]
    start_date = datetime(2023, 1, 1)
    records = []

    for _ in range(n):
        date = start_date + timedelta(days=random.randint(0, 540))
        records.append({
            "store_id": random.choice(stores),
            "sku": random.choice(skus),
            "units_sold": random.randint(1, 20),
            "price_per_unit": round(random.uniform(1.5, 25.0), 2),
            "date": date.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"POS data saved to {output_path}")


def simulate_customers(n=5000, output_path="data/raw/customers.csv"):
    segments = ["Gold", "Silver", "Bronze"]
    genders = ["Male", "Female"]
    countries = ["UAE", "KSA", "Qatar", "Oman"]
    records = []

    for i in range(n):
        join_date = datetime(2021, 1, 1) + timedelta(days=random.randint(0, 900))
        records.append({
            "customer_id": f"CUST_{i}",
            "gender": random.choice(genders),
            "age": random.randint(18, 65),
            "country": random.choice(countries),
            "segment": random.choice(segments),
            "join_date": join_date.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Customers data saved to {output_path}")


def simulate_loyalty_txns(n=25000, output_path="data/raw/loyalty_transactions.csv"):
    cust_ids = [f"CUST_{i}" for i in range(5000)]
    records = []

    for _ in range(n):
        txn_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 540))
        records.append({
            "customer_id": random.choice(cust_ids),
            "points_earned": random.randint(10, 500),
            "txn_amount": round(random.uniform(10.0, 300.0), 2),
            "txn_date": txn_date.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Loyalty transactions saved to {output_path}")


def main():
    os.makedirs("data/raw", exist_ok=True)
    simulate_pos_data()
    simulate_customers()
    simulate_loyalty_txns()


if __name__ == "__main__":
    main()
