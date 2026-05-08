import pandas as pd
import numpy as np
import random

np.random.seed(42)

names = ["Ali","Ahmed","Sara","Fatima","Usman","Ayesha","Bilal","Hina","Zain","Hamza"]

data = []

for i in range(10000):

    name = random.choice(names) + "_" + str(i)

    # CGPA (normal distribution)
    cgpa = round(np.clip(np.random.normal(3.0, 0.5), 2.0, 4.0), 2)

    # Skill pattern types
    profile_type = random.choice(["balanced", "specialist", "weak"])

    if profile_type == "balanced":
        skills = np.random.randint(5, 9, size=7)

    elif profile_type == "specialist":
        skills = np.random.randint(2, 6, size=7)
        strong_index = random.randint(0,6)
        skills[strong_index] = random.randint(8,10)

    else:  # weak
        skills = np.random.randint(1, 5, size=7)

    row = [name] + list(skills) + [cgpa]
    data.append(row)

df = pd.DataFrame(data, columns=[
    "Student","Python","ML","Web","AI",
    "Communication","Leadership","Database","CGPA"
])

df.to_csv("students_skills_10k.csv", index=False)

print("✅ 10K dataset created!")