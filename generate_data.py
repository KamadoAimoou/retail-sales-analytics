"""Generate categories and products tables (synthetic data, prices in KZT)."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  
OUT = Path("data")
OUT.mkdir(exist_ok=True)

# categories
category_names = ["Dairy", "Bakery", "Beverages", "Snacks",
                  "Meat", "Fruits & Vegetables", "Household", "Sweets"]
categories = pd.DataFrame({
    "category_id": range(1, len(category_names) + 1),
    "category_name": category_names,
})

# products: (name, category, price)
catalog = [
    ("Milk 1L", "Dairy", 480), ("Kefir 1L", "Dairy", 450),
    ("Butter 180g", "Dairy", 1250), ("Yogurt 350g", "Dairy", 390),
    ("White bread", "Bakery", 160), ("Lavash", "Bakery", 230),
    ("Croissant", "Bakery", 320), ("Samsa", "Bakery", 450),
    ("Water 1.5L", "Beverages", 190), ("Black tea 100 bags", "Beverages", 1350),
    ("Cola 1.5L", "Beverages", 520), ("Orange juice 1L", "Beverages", 780),
    ("Potato chips 140g", "Snacks", 780), ("Salted nuts 150g", "Snacks", 990),
    ("Crackers 200g", "Snacks", 430), ("Sunflower seeds 300g", "Snacks", 520),
    ("Chicken breast 1kg", "Meat", 2400), ("Beef 1kg", "Meat", 4300),
    ("Sausage 400g", "Meat", 1650), ("Eggs (10 pcs)", "Meat", 780),
    ("Potatoes 1kg", "Fruits & Vegetables", 260), ("Onions 1kg", "Fruits & Vegetables", 190),
    ("Tomatoes 1kg", "Fruits & Vegetables", 690), ("Apples 1kg", "Fruits & Vegetables", 590),
    ("Dish soap 500ml", "Household", 650), ("Laundry powder 3kg", "Household", 2900),
    ("Toilet paper (8 rolls)", "Household", 1450), ("Trash bags (30 pcs)", "Household", 520),
    ("Chocolate bar 90g", "Sweets", 540), ("Candy mix 500g", "Sweets", 1350),
    ("Cookies 300g", "Sweets", 580), ("Honey 500g", "Sweets", 2400),
]

category_id = dict(zip(categories.category_name, categories.category_id))

products = pd.DataFrame(catalog, columns=["product_name", "category_name", "unit_price"])
products["category_id"] = products["category_name"].map(category_id)
products["unit_cost"] = (products["unit_price"] * rng.uniform(0.60, 0.85, len(products))).round()
products.insert(0, "product_id", range(1, len(products) + 1))
products = products[["product_id", "product_name", "category_id", "unit_price", "unit_cost"]]

# ---------- customers ----------
N_CUST = 900
signup = pd.Timestamp("2024-06-01") + pd.to_timedelta(rng.integers(0, 580, N_CUST), unit="D")
customers = pd.DataFrame({
    "customer_id": range(1, N_CUST + 1),
    "signup_date": signup.date,
    "loyalty_member": rng.random(N_CUST) < 0.45,
})
# some customers shop much more often than others
cust_weight = rng.lognormal(0, 0.9, N_CUST)
cust_weight /= cust_weight.sum()

# ---------- how many orders per day ----------
days = pd.date_range("2025-01-01", "2025-12-31")
month_mult = {1: .92, 2: .95, 3: 1.12, 4: .98, 5: 1.0, 6: .97,
              7: .93, 8: .95, 9: 1.02, 10: 1.0, 11: 1.05, 12: 1.25}   # March = Nauryz, Dec = New Year
dow_mult = {0: .85, 1: .85, 2: .9, 3: .95, 4: 1.1, 5: 1.35, 6: 1.25}  # weekends are busier
lam = [14 * month_mult[d.month] * dow_mult[d.dayofweek] for d in days]
orders_per_day = rng.poisson(lam)

# ---------- popularity of products and hours ----------
pop = 1 / products["unit_price"].to_numpy() ** 0.55      # cheap items are bought more often
pop /= pop.sum()
hour_p = np.array([0]*8 + [1, 3, 5, 6, 6, 5, 4, 5, 7, 9, 10, 8, 4] + [0]*3, dtype=float)
hour_p /= hour_p.sum()
payments = ["Kaspi QR", "Card", "Cash"]

# ---------- orders and order_items ----------
order_rows, item_rows = [], []
order_id = 0
for day, n in zip(days, orders_per_day):
    for _ in range(n):
        order_id += 1
        cust = int(rng.choice(N_CUST, p=cust_weight)) + 1
        loyal = bool(customers.loyalty_member[cust - 1])
        known = loyal or rng.random() < 0.35              # other buyers are often anonymous
        order_rows.append({
            "order_id": order_id,
            "customer_id": cust if known else None,
            "order_date": day.date(),
            "order_time": f"{rng.choice(24, p=hour_p):02d}:{rng.integers(0, 60):02d}:00",
            "payment_method": rng.choice(payments, p=[0.5, 0.25, 0.25]),
        })
        n_items = min(1 + rng.poisson(2.2), 9)
        for p in rng.choice(len(products), size=n_items, replace=False, p=pop):
            item_rows.append({
                "order_id": order_id,
                "product_id": int(p) + 1,
                "quantity": 1 + int(rng.poisson(0.5)),
                "discount_pct": 5 if (loyal and rng.random() < 0.5) else 0,
            })

orders = pd.DataFrame(order_rows)
orders["customer_id"] = orders["customer_id"].astype("Int64")   # integers with NULLs
order_items = pd.DataFrame(item_rows)
order_items.insert(0, "item_id", range(1, len(order_items) + 1))

customers.to_csv(OUT / "customers.csv", index=False)
orders.to_csv(OUT / "orders.csv", index=False)
order_items.to_csv(OUT / "order_items.csv", index=False)
print(f"customers: {len(customers)}, orders: {len(orders)}, order items: {len(order_items)}")


categories.to_csv(OUT / "categories.csv", index=False)
products.to_csv(OUT / "products.csv", index=False)

print(f"categories: {len(categories)}, products: {len(products)}")