"""Generate categories and products tables (synthetic data, prices in KZT)."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # fixed seed -> same result every run
OUT = Path("data")
OUT.mkdir(exist_ok=True)

# ---------- categories ----------
category_names = ["Dairy", "Bakery", "Beverages", "Snacks",
                  "Meat", "Fruits & Vegetables", "Household", "Sweets"]
categories = pd.DataFrame({
    "category_id": range(1, len(category_names) + 1),
    "category_name": category_names,
})

# ---------- products: (name, category, price) ----------
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

# ---------- save ----------
categories.to_csv(OUT / "categories.csv", index=False)
products.to_csv(OUT / "products.csv", index=False)

print(f"categories: {len(categories)}, products: {len(products)}")