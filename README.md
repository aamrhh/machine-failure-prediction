# Machine Failure Early Warning

The goal is to find machines that will fail in the next 7 days, so the team can check them before they break.

**Short answer:** the model finds the same number of broken machines as a simple rule, with about **half the visits**. It is **not ready to use yet**, but it is worth developing further.

---

## 1. How to run

You need **Python 3.11**.

```bash
# 1. Download the project
git clone https://github.com/aamrhh/machine-failure-prediction.git
cd machine-failure-prediction

# 2. Make a virtual environment and turn it on
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt
```

Then open the notebooks in VS Code (with the Jupyter extension) or in Jupyter Notebook. Run them **in this order**, using **Restart and Run All**:

| Order | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/01_explore_data.ipynb` | Explores and cleans the data. Saves `data/machine_observations_clean.csv` |
| 2 | `notebooks/02_leakage_check.ipynb` | Checks which columns can be trusted |
| 3 | `notebooks/03_model.ipynb` | Builds and tests the model, and gives the recommendation |

Notebook 01 must run first. Notebooks 02 and 03 use the clean file it saves.

## 2. Folders

```
machine-failure-prediction/
├── data/
│   ├── machine_observations.csv         original data
│   └── machine_observations_clean.csv   made by Notebook 01
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_leakage_check.ipynb
│   └── 03_model.ipynb
├── requirements.txt
└── README.md
```

---

## 3. The data

- 10,414 readings from **120 machines** at **3 sites**
- From **Jan 2024 to Feb 2025**
- Each machine is read about every **4 days**
- Only about **5%** of readings have a failure in the next 7 days. Failures are rare.

## 4. Data problems found and fixed (Notebook 01)

| Problem | What was found | Fix |
|---|---|---|
| **South pressure in the wrong unit** | From 14 Dec 2024, all 399 South readings dropped from about 501 to 73. 73 × 6.895 = 504, which is the normal value in kPa. | Treated as psi. Converted back to kPa. |
| **Vibration of −999** | 10 readings were exactly −999. A machine cannot shake a negative amount. | Treated as missing. |
| **Duplicate rows** | 80 rows were copies (same machine, same time, same values). | Removed. 10,414 rows became 10,334. |
| **"unknown" machine class** | 15 rows. Each machine has only one class in its other rows. | Filled with that machine's class. |

Missing values are filled later, in Notebook 03, using the **median of the training data only**. This stops information from the test data leaking in.

## 5. Leakage check (Notebook 02)

Each reading must be scored using only what is known **at that moment**. Sensor readings are fine, because they are measured at that moment. But three columns come from records written by people, so the timing is not clear.

| Column | Decision | Why |
|---|---|---|
| `corrective_orders_14d` | **Not used** | When there is a repair order, 44.4% of readings have a failure coming. When there is no repair order, only 1.2% do. Also, 71.4% of the 339 failures already show a repair order before them. If the team knew about 7 in 10 problems early, the machines should not keep breaking. So the repair order is probably written **because of** the breakdown. |
| `inspection_interval_days` | **Not used** | It is always 14 days when there is a repair order. So it carries the same information. |
| `unplanned_stops_14d` | **Used, but needs checking** | More stops slowly means more failures (0.9% to 21.8%). This looks like a real warning sign, so it is used in the model. But stops are also written by people, like repair orders, so their timing should still be confirmed (Section 6). |

## 6. Assumptions and questions for the data owner

The data alone cannot prove these. So each one is written down as an assumption, with the question that would confirm it.

| Assumption | Question for the data owner | If yes |
|---|---|---|
| Repair orders may be written because of the breakdown | Is a repair order entered on the day it is raised, with that day's date? | Repair orders can be used. The model could need far fewer visits (141 instead of 376). |
| The inspection interval changes because of a repair order | Is the inspection interval the value in force at the time of the reading? | The inspection interval can be used. |
| Unplanned stops are recorded on the day they happen | Are unplanned stops entered on the day they happen? | The model can keep using this column safely. |
| South pressure from 14 Dec 2024 is in psi | Did the South site change its pressure unit to psi in Dec 2024? | The psi to kPa fix is correct. |
| −999 in vibration means "no reading" | Is −999 a code for a missing vibration reading? | Treating −999 as missing is correct. |

---

## 7. How the model was tested (Notebook 03)

**Time-based split.**
- **Train:** Jan to Oct 2024 (7,238 readings)
- **Test:** Nov 2024 to Feb 2025 (2,922 readings, 151 failures)
- The readings from 25 to 31 Oct 2024 are **not used** (a 7-day gap).

**Why split by time?** The model will be used on future readings. So it is trained on older months and tested on newer months. A random split would be unfair. Readings from the same machine a few days apart look almost the same. With a random split, one could end up in training and the other in testing. The model would then be tested on something it has almost already seen, so the test score would look better than it really is.

**Why the 7-day gap?** The target looks 7 days ahead. A training reading on 30 Oct would look ahead to 6 Nov, which is in the test period. The gap stops training data from seeing into the test period.

**Why not accuracy?**
A model that always says "no failure" finds zero broken machines, but still gets about 95% accuracy. So two other measures are used:
- **Recall:** of all broken machines, how many are found?
- **Precision:** of all visits, how many find a broken machine?

**How methods are compared.**
1. Each method sorts the test readings, most risky first.
2. The team visits from the top of the list, until **70% of broken machines (106 of 151)** are found. This is the recall goal.
3. Count the visits. **Fewer visits = better.**

**Why 70%?**
Missing a failure is costly, so the goal is to find most broken machines. But finding almost all of them takes many more visits (90% needs 812 visits, more than double). 70% is a middle point. Other levels are shown in Section 8, so the team can choose.

**Baseline (simple rules to beat).** A model is only useful if it beats a simple rule. So each sensor was tried alone, on the training data, as a simple rule: sort the readings by that sensor and visit from the top. Two sensors worked best, and were almost tied:
- **Lowest pressure first**
- **Highest vibration first**

The model must need fewer visits than both.

**Model.** **Logistic regression** gives points to each clue (for example, more vibration or lower pressure means more points). It learns the points from the training data, then gives every reading a risk score. A **random forest** was also tested, to see if a more complex model helps.

## 8. Results

**Visits needed to find 106 of the 151 broken machines (test data, 17 weeks)**

| How machines are picked | Visits | Wasted visits (machine was fine) |
|---|---|---|
| Lowest pressure first (simple rule) | 699 | 593 |
| Highest vibration first (simple rule) | 754 | 648 |
| **Logistic regression (main model)** | **376** | **270** |
| Random forest | 401 | 295 |
| With untrusted columns | 141 (not trusted) | 35 |

All methods have the same goal: find 106 broken machines. So the method with **fewer visits** is better.

- The main model needs **about half the visits** of the simple rules.
- **Random forest was not better.** The simpler logistic regression is kept, because it is easier to explain.
- **With the untrusted columns, only 141 visits are needed.** This is not trusted, because those columns may be written because of the breakdown. A high score is not always a good result.

**How many visits should the team make? (main model, totals over 17 weeks)**

| Goal | Visits | Visits per week | Found | Missed | Wasted visits |
|---|---|---|---|---|---|
| Find 50% of broken machines (76 of 151) | 195 | 11.5 | 76 | 75 | 119 |
| Find 60% (91 of 151) | 283 | 16.6 | 91 | 60 | 192 |
| **Find 70% (106 of 151)** | **376** | **22.1** | **106** | **45** | **270** |
| Find 80% (121 of 151) | 524 | 30.8 | 121 | 30 | 403 |
| Find 90% (136 of 151) | 812 | 47.8 | 136 | 15 | 676 |

More visits find more broken machines, but also more machines that are fine. The team should choose based on how many visits they can handle.

---

## 9. Recommendation

**Not ready to use yet, but worth developing further.**

Why not yet:
- **Too many wasted visits.** Even with the model, 270 of 376 visits (about 7 in 10) find a machine that is fine. The team may stop trusting the alerts.
- **Two columns cannot be used yet.** It is not known if repair orders and inspection intervals are in the system at the time of the reading. Only the data owner can confirm this.
- **The data had errors.** New data could have the same errors (unit change, −999, duplicates).

Next steps:
1. **Ask the data owner** when repair orders, stops and failures are entered into the system. This decides if those columns can be used.
2. **If the columns are safe,** test the model again with them. They could greatly reduce wasted visits.
3. **Try the model at one site,** and compare its alerts with what the team really finds.
4. **Add automatic data checks** for new readings (for example, a sudden pressure drop, or values like −999), so errors are caught before the model uses them.

---

Author: [aamrhh](https://github.com/aamrhh)