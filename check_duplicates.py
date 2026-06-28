import pandas as pd

data = pd.read_csv("IFND.csv", encoding="latin1")

my_headlines = [
    "India's Chandrayaan-3 mission successfully landed near the Moon's south pole in August 2023",
    "India surpassed China to become the world's most populous country in 2023",
    "India won the ICC T20 World Cup in 2024",
    "The Statue of Unity, the world's tallest statue, is located in Gujarat",
    "India commissioned its first indigenous aircraft carrier, INS Vikrant, in 2022",
    "India's UPI (Unified Payments Interface) is one of the largest digital payment systems globally",
    "The Reserve Bank of India is headquartered in Mumbai",
    "The Kumbh Mela is recognized as one of the largest peaceful gatherings of people on Earth",
    "India developed and tested its indigenous light combat aircraft, the Tejas",
    "The Taj Mahal in Agra was completed in the 17th century under Mughal emperor Shah Jahan"
]

# Check exact matches
for h in my_headlines:
    match = data[data["Statement"].str.strip().str.lower() == h.strip().lower()]
    if len(match) > 0:
        print(f"EXACT MATCH FOUND: {h}")
    else:
        print(f"Not found: {h}")